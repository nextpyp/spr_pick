from __future__ import annotations


import torch
import torch.nn as nn
import spr_pick

from torch import Tensor

from spr_pick.params import (
    ConfigValue,
    PipelineOutput,
    Pipeline,
    NoiseValue,
    Loss,
)
import numpy as np

from scipy.signal import medfilt
from spr_pick.models import NoiseNetwork
from spr_pick.datasets import NoisyDataset
from spr_pick.models import JointNetwork, Detector
from spr_pick.models import ResNet6, ResNet8, ResNet16
from spr_pick.models import LinearClassifier
from spr_pick.datasets import DetectionDataset
from spr_pick.models import NoiseEstNetwork

from typing import Dict, List
from spr_pick.utils.losses import js_div_loss_2d
from spr_pick.utils.losses import PuLoss, modified_pu_loss

class Denoiser(nn.Module):

    MODEL = "denoiser_model"
    SIGMA_ESTIMATOR = "sigma_estimation_model"
    ESTIMATED_SIGMA = "estimated_sigma"
    PROB_ESTIMATOR = "detector_model"

    def __init__(
        self, cfg: Dict, device: str = None, mode: str = None
    ):
        super().__init__()
        # Configure device
        if device:
            device = torch.device(device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Store the denoiser configuration
        self.cfg = cfg
        self.mode = mode
        # Models to use during training, these can be parallelised
        self.models = nn.ModuleDict()
        # References to models that are guaranteed to be device independent
        self._models = nn.ModuleDict()
        # Initialise networks using current configuration
        self.init_networks()
        # Learnable parameters
        self.l_params = nn.ParameterDict()
        self.init_l_params()

    def init_networks(self):
        # Calculate input and output channel count for networks
        in_channels = self.cfg[ConfigValue.IMAGE_CHANNELS]
        if self.cfg[ConfigValue.PIPELINE] == Pipeline.SSDN:
            if self.cfg[ConfigValue.DIAGONAL_COVARIANCE]:
                out_channels = in_channels * 2  # Means, diagonal of A
            else:
                out_channels = (
                    in_channels + (in_channels * (in_channels + 1)) // 2
                )  # Means, triangular A.
        else:
            out_channels = in_channels

        # Create general denoising model
        # self.add_model(
        #     Denoiser.MODEL,
        #     NoiseNetwork(
        #         in_channels=in_channels,
        #         out_channels=out_channels,
        #         blindspot=self.cfg[ConfigValue.BLINDSPOT],
        #     ),
        # )
        self.add_model(
            Denoiser.MODEL, 
            JointNetwork(
                in_channels=in_channels,
                out_channels=out_channels,
                blindspot=self.cfg[ConfigValue.BLINDSPOT],
                detect=True,
            ),
        )
        # Create separate model for variable parameter estimation
        if (
            self.cfg[ConfigValue.PIPELINE] == Pipeline.SSDN
            and self.cfg[ConfigValue.NOISE_VALUE] == NoiseValue.UNKNOWN_VARIABLE
        ):
            # self.add_model(
            #     Denoiser.SIGMA_ESTIMATOR,
            #     NoiseNetwork(
            #         in_channels=in_channels,
            #         out_channels=1,
            #         blindspot=False,
            #         zero_output_weights=True,
            #     ),
            # )
            #test Topaz 
            # self.add_model(
            #     Denoiser.SIGMA_ESTIMATOR, 
            #     LinearClassifier(ResNet8(units=64, bn=False)),
            #     )
            # print('width')
            # print(LinearClassifier(ResNet8(units=64, bn=False)).width)
            self.add_model(
                Denoiser.SIGMA_ESTIMATOR,
                NoiseEstNetwork(
                    in_channels = in_channels,
                    out_channels = 1,
                    blindspot=False,
                    detect=False,
                ),
            )
            self.add_model(
                Denoiser.PROB_ESTIMATOR,
                Detector(),
                )
            # self.add_model(
            #     Denoiser.SIGMA_ESTIMATOR,
            #     JointNetwork(
            #         in_channels = in_channels,
            #         out_channels = 1,
            #         blindspot=False,
            #         detect=False,
            #     ),
            # )

    def fill(self, stride=1):
        return self.models[Denoiser.PROB_ESTIMATOR].fill(stride=stride)

    def unfill(self):
        return self.models[Denoiser.PROB_ESTIMATOR].unfill()

    def init_l_params(self):
        if (
            self.cfg[ConfigValue.PIPELINE] == Pipeline.SSDN
            and self.cfg[ConfigValue.NOISE_VALUE] == NoiseValue.UNKNOWN_CONSTANT
        ):
            init_value = torch.zeros((1, 1, 1, 1))
            self.l_params[Denoiser.ESTIMATED_SIGMA] = nn.Parameter(init_value)

    def get_model(self, model_id: str, parallelised: bool = True) -> nn.Module:
        model_dict = self.models if parallelised else self._models
        return model_dict[model_id]

    def add_model(self, model_id: str, model: nn.Module, parallelise: bool = False):
        self._models[model_id] = model
        if parallelise:
            parallel_model = nn.DataParallel(model)
        else:
            parallel_model = model
        # Move to master device (GPU 0 or CPU)
        parallel_model.to(self.device)
        self.models[model_id] = parallel_model

    def forward(self, data: Tensor) -> Tensor:
        """Pass an input into the denoiser for inference. This will not train
        the network. Inference will be applied using current model state with
        the configured pipeline.

        Args:
            data (Tensor): Image or batch of images to denoise in BCHW format.

        Returns:
            Tensor: Denoised image or images.
        """
        assert NoisyDataset.INPUT == 0
        inputs = [data]
        outputs = self.run_pipeline(inputs)
        return outputs[PipelineOutput.IMG_DENOISED]

    def run_pipeline(self, data: List, alpha = 0, pi = 0, train = True, **kwargs):
        if self.cfg[ConfigValue.PIPELINE] == Pipeline.MSE and self.mode == "denoise":
            return self._mse_pipeline(data, **kwargs)
        elif self.cfg[ConfigValue.PIPELINE] == Pipeline.SSDN and self.mode == "denoise":
            return self._ssdn_pipeline(data, **kwargs)
        elif self.cfg[ConfigValue.PIPELINE] == Pipeline.MASK_MSE and self.mode == "denoise":
            return self._mask_mse_pipeline(data, **kwargs)
            # return self._mse_pipeline(data, **kwargs)
        elif self.mode == "detect":
            return self._detect_only_pipeline(data, **kwargs)
        elif self.mode == "ssdet":
            return self._ss_detect_only_pipeline(data, **kwargs)
        elif self.mode == "joint":
            return self._new_pipeline(data, alpha, train = train, **kwargs)

        else:
            raise NotImplementedError("Unsupported processing pipeline")

    def _mse_pipeline(self, data: List, **kwargs) -> Dict:
        outputs = {PipelineOutput.INPUTS: data}
        # Run the input through the model
        inp = data[NoisyDataset.INPUT].to(self.device)
        inp.requires_grad = True
        cleaned = self.models[Denoiser.MODEL](inp)
        outputs[PipelineOutput.IMG_DENOISED] = cleaned

        # If reference images are provided calculate the loss
        # as MSE whilst preserving individual loss per batch
        if len(data) >= NoisyDataset.REFERENCE:
            ref = data[NoisyDataset.REFERENCE].to(self.device)
            ref.requires_grad = True
            loss = nn.MSELoss(reduction="none")(cleaned, ref)
            loss = loss.view(loss.shape[0], -1).mean(1, keepdim=True)
            outputs[PipelineOutput.LOSS] = loss

        return outputs

    def _mask_mse_pipeline(self, data: List, **kwargs) -> Dict:
        outputs = {PipelineOutput.INPUTS: data}
        # Run the input through the model
        inp = data[NoisyDataset.INPUT].to(self.device)
        inp.requires_grad = True
        cleaned = self.models[Denoiser.MODEL](inp)
        outputs[PipelineOutput.IMG_DENOISED] = cleaned

        # If reference images are provided calculate the loss
        # as MSE whilst preserving individual loss per batch
        if len(data) >= NoisyDataset.REFERENCE:
            ref = data[NoisyDataset.REFERENCE].to(self.device)
            ref.requires_grad = True

            if NoisyDataset.Metadata.MASK_COORDS in data[NoisyDataset.METADATA]:
                mask_coords = data[NoisyDataset.METADATA][NoisyDataset.Metadata.MASK_COORDS]
                loss = spr_pick.utils.n2v_loss.loss_mask_mse(mask_coords, cleaned, ref)
                loss = loss.view(loss.shape[0], -1).mean(1, keepdim=True)
                outputs[PipelineOutput.LOSS] = loss


        return outputs

    def _ss_detect_only_pipeline(self, data:List, **kwargs) -> Dict:
        debug = False
        if len(data) > 3:
            inp, target, inp_aug = data[DetectionDataset.INPUT], data[DetectionDataset.TARGET], data[DetectionDataset.AUG_INP]
            metadata = data[DetectionDataset.METADATA]
            # inps = torch.cat((inp, inp_aug), 0)
            # print('inp_aug', inp_aug)
            # print('concatenated')
            # print(inp.shape)
            # print(inps.shape)
            target = target.to(self.device)
            # print('target',target)
            inp = inp.to(self.device)
            inp_aug = inp_aug.to(self.device)
            # print(inp[0])
            # print(inp_aug[0])
            input_shape = metadata[DetectionDataset.Metadata.IMAGE_SHAPE]
            num_channels = self.cfg[ConfigValue.IMAGE_CHANNELS]
            assert num_channels in [1, 3]

            diagonal_covariance = self.cfg[ConfigValue.DIAGONAL_COVARIANCE]
            if debug:
                print("Image shape:", input_shape)
            if debug:
                print("Num. channels:", num_channels)

            # Clean data distribution.
            # Calculation still needed for line 175
            num_output_components = (
                num_channels + (num_channels * (num_channels + 1)) // 2
            )  # Means, triangular A.
            if diagonal_covariance:
                num_output_components = num_channels * 2  # Means, diagonal of A.
            if debug:
                print("Num. output components:", num_output_components)
            # Call the NN with the current image etc.
            det_orig, net_out = self.models[Denoiser.MODEL](inp)
            det_aug, net_aug_out = self.models[Denoiser.MODEL](inp_aug)
            criteria_sup = nn.BCEWithLogitsLoss()
            criteria_cons = nn.MSELoss()
            criteria_kl = nn.KLDivLoss()
            sig = nn.Sigmoid()

            
            # print('sigmoid')
            # print(det_orig_sig)
            # print(neg_sig_orig)
            # print(det_orig_sig.shape)

            # det_orig = det[:det.shape[0]//2]
            # det_aug = det[det.shape[0]//2:]
            # print('det', det)
            # print('det_orig', det_orig)
            # print('det_aug', det_aug)
            det_orig_f = det_orig.view(det_orig.shape[0],-1)
            det_aug_f = det_aug.view(det_aug.shape[0], -1)
            det_orig_sig = sig(det_orig_f)
            det_aug_sig = sig(det_aug_f)
            # frac = torch.log(det_orig_sig / det_aug_sig)
            # # print(frac)
            # sum_kl = torch.sum(det_orig_sig * frac)
            # print('sum_kl', sum_kl)
            # neg_sig_orig = 1 - det_orig_sig
            # neg_sig_aug = 1 - det_aug_sig 
            # det_orig_all = torch.cat((torch.log(det_orig_sig), torch.log(neg_sig_orig)), 1)
            # det_orig_all_nonlog = torch.cat((det_orig_sig, neg_sig_orig), 1)
            # det_aug_all = torch.cat((det_aug_sig, neg_sig_aug), 1)
            # print('det_orig_all')
            # # print(det_orig_all.shape)
            # print(det_orig_all)
            # print(det_orig_all_nonlog)
            # print(det_orig_f)
            # # print()
            # print('det_aug_all')
            # print(det_aug_all)
            if target.shape[-1] > 1:
                target = target.view(target.shape[0],-1)
            # print('2 target', target[target == 2])
            # print(target)
            det_orig_f_zero = det_orig_f[target == 0]
            det_orig_f_one = det_orig_f[target == 1]
            # print(det_orig_f_zero.shape[0])
            # print(det_orig_f_zero.shape[0]>0)
            if det_orig_f_one.shape[0] > 0 and det_orig_f_zero.shape[0] > 0:

                supervised_loss = criteria_sup(det_orig_f_zero, target[target == 0]) + criteria_sup(det_orig_f_one, target[target == 1])
            elif det_orig_f_one.shape[0] > 0 and det_orig_f_zero.shape[0] == 0:
                supervised_loss = criteria_sup(det_orig_f_one, target[target == 1])
            elif det_orig_f_one.shape[0] == 0 and det_orig_f_zero.shape[0] > 0:
                supervised_loss = criteria_sup(det_orig_f_zero, target[target == 0])
            else:
                supervised_loss = 0

            # consistency_loss = js_div_loss_2d(det_orig, det_aug)
            consistency_loss = criteria_cons(det_orig_sig, det_aug_sig)
            # consistency_kl = criteria_kl(det_orig_all, det_aug_all)
            # consistency_kl = sum_kl
            total_loss = supervised_loss + 0.01*consistency_loss
            # print('consistency_kl', consistency_loss)
            # print('supervised_loss loss', supervised_loss)

            total_loss = total_loss
            # print(total_loss)
            mu_x = net_out[:, 0:num_channels, ...]  # Means (NCHW).
            A_c = net_out[
                :, num_channels:num_output_components, ...
            ] 
            det = det_orig

        else:
            inp, target = data[DetectionDataset.INPUT], data[DetectionDataset.TARGET]
            noisy_in = inp.to(self.device)
            target = target.to(self.device)
            metadata = data[DetectionDataset.METADATA]
            #noise_params_in = metadata[NoisyDataset.Metadata.INPUT_NOISE_VALUES]

            # config for noise params/style
            # noise_style = self.cfg[ConfigValue.NOISE_STYLE]
            noise_params = self.cfg[ConfigValue.NOISE_VALUE]

            # Equivalent of blindspot_pipeline
            input_shape = metadata[DetectionDataset.Metadata.IMAGE_SHAPE]
            num_channels = self.cfg[ConfigValue.IMAGE_CHANNELS]
            assert num_channels in [1, 3]

            diagonal_covariance = self.cfg[ConfigValue.DIAGONAL_COVARIANCE]
            if debug:
                print("Image shape:", input_shape)
            if debug:
                print("Num. channels:", num_channels)

            # Clean data distribution.
            # Calculation still needed for line 175
            num_output_components = (
                num_channels + (num_channels * (num_channels + 1)) // 2
            )  # Means, triangular A.
            if diagonal_covariance:
                num_output_components = num_channels * 2  # Means, diagonal of A.
            if debug:
                print("Num. output components:", num_output_components)
            # Call the NN with the current image etc.
            det, net_out = self.models[Denoiser.MODEL](noisy_in)
            criteria = nn.BCEWithLogitsLoss(reduction="none")
            det_f = det.view(det.shape[0],-1)
            # print('det')
            # print(det.shape)
            # print('target')
            # print(target.shape)
            if target.shape[-1] > 1:
                target = target.view(target.shape[0],-1)
            total_loss = criteria(det_f, target)
            # print('det loss')
            # print(det_loss)
            # print(det_loss.shape)
            # print('model')
            # print(self.models[Denoiser.MODEL])
            # net_out = net_out.type(torch.float64)
            # print('net out')
            # print(net_out.shape)

            if debug:
                print("Net output shape:", net_out.shape)
            mu_x = net_out[:, 0:num_channels, ...]  # Means (NCHW).
            A_c = net_out[
                :, num_channels:num_output_components, ...
            ] 
        return {
        PipelineOutput.INPUTS: data,
        PipelineOutput.TARGET: target,
        PipelineOutput.DETECT: det,
        PipelineOutput.LOSS: total_loss,
        PipelineOutput.IMG_MU: mu_x
        }



    def _detect_only_pipeline(self, data: List, **kwargs) -> Dict:
        debug =False
        inp, target = data[DetectionDataset.INPUT], data[DetectionDataset.TARGET]
        noisy_in = inp.to(self.device)
        target = target.to(self.device)
        metadata = data[DetectionDataset.METADATA]
        #noise_params_in = metadata[NoisyDataset.Metadata.INPUT_NOISE_VALUES]

        # config for noise params/style
        # noise_style = self.cfg[ConfigValue.NOISE_STYLE]
        noise_params = self.cfg[ConfigValue.NOISE_VALUE]

        # Equivalent of blindspot_pipeline
        input_shape = metadata[DetectionDataset.Metadata.IMAGE_SHAPE]
        num_channels = self.cfg[ConfigValue.IMAGE_CHANNELS]
        assert num_channels in [1, 3]

        diagonal_covariance = self.cfg[ConfigValue.DIAGONAL_COVARIANCE]
        if debug:
            print("Image shape:", input_shape)
        if debug:
            print("Num. channels:", num_channels)

        # Clean data distribution.
        # Calculation still needed for line 175
        num_output_components = (
            num_channels + (num_channels * (num_channels + 1)) // 2
        )  # Means, triangular A.
        if diagonal_covariance:
            num_output_components = num_channels * 2  # Means, diagonal of A.
        if debug:
            print("Num. output components:", num_output_components)
        # Call the NN with the current image etc.
        det, net_out = self.models[Denoiser.MODEL](noisy_in)
        criteria = nn.BCEWithLogitsLoss(reduction="none")
        det_f = det.view(det.shape[0],-1)
        # print('det')
        # print(det.shape)
        # print('target')
        # print(target.shape)
        if target.shape[-1] > 1:
            target = target.view(target.shape[0],-1)
        det_loss = criteria(det_f, target)
        # print('det loss')
        # print(det_loss)
        # print(det_loss.shape)
        # print('model')
        # print(self.models[Denoiser.MODEL])
        # net_out = net_out.type(torch.float64)
        # print('net out')
        # print(net_out.shape)

        if debug:
            print("Net output shape:", net_out.shape)
        mu_x = net_out[:, 0:num_channels, ...]  # Means (NCHW).
        A_c = net_out[
            :, num_channels:num_output_components, ...
        ] 

        param_est_net_out = self.models[Denoiser.SIGMA_ESTIMATOR](noisy_in)
        
        return {
        PipelineOutput.INPUTS: data,
        PipelineOutput.TARGET: target,
        PipelineOutput.DETECT: det,
        PipelineOutput.LOSS: det_loss,
        PipelineOutput.IMG_MU: mu_x
        }

    def _new_hm_pipeline(self, data: List, alpha: float, train: bool, **kwargs) -> Dict:
        debug = False
        noise_style = "gauss"
        noise_params = self.cfg[ConfigValue.NOISE_VALUE]
        # test_data = None
        if len(data) > 3:
            #target should all be one, only sample labeled ones 
            inp, target, hm = data[DetectionDataset.INPUT], data[DetectionDataset.TARGET], data[DetectionDataset.HM]
            metadata = data[DetectionDataset.METADATA]
            gt = metadata[DetectionDataset.Metadata.GT]
            # inps = torch.cat((inp, inp_aug), 0)
            # print('inp_aug', inp_aug)
            # print('concatenated')
            # print(inp.shape)
            ind = metadata[DetectionDataset.Metadata.INDEXES]
            # print('ind', ind)
            # print(inps.shape)
            target = target.to(self.device)
            # print('target',target)
            inp = inp.to(self.device)
            hm = hm.to(self.device)
        return 

    def _new_pipeline(self, data: List, alpha: float, train: bool, **kwargs) -> Dict:
        debug = False
        noise_style = "gauss"
        noise_params = self.cfg[ConfigValue.NOISE_VALUE]
        # test_data = None
        if len(data) > 2:
            #target should all be one, only sample labeled ones 
            inp, target = data[DetectionDataset.INPUT], data[DetectionDataset.TARGET]
            # print('inp', inp.shape)
            # inp = inp[:,:,:64,:64]
            metadata = data[DetectionDataset.METADATA]
            gt = metadata[DetectionDataset.Metadata.GT]
            # inps = torch.cat((inp, inp_aug), 0)
            # print('inp_aug', inp_aug)
            # print('concatenated')
            # print(inp.shape)
            ind = metadata[DetectionDataset.Metadata.INDEXES]
            # print('ind', ind)
            # print(inps.shape)
            target = target.to(self.device)
            # print('target',target)
            inp = inp.to(self.device)
            # inp_aug = inp_aug.to(self.device)
            noisy_in = inp
            input_shape = metadata[DetectionDataset.Metadata.IMAGE_SHAPE]
            num_channels = self.cfg[ConfigValue.IMAGE_CHANNELS]
            assert num_channels in [1, 3]

            diagonal_covariance = self.cfg[ConfigValue.DIAGONAL_COVARIANCE]
            if debug:
                print("Image shape:", input_shape)
            if debug:
                print("Num. channels:", num_channels)

            # Clean data distribution.
            # Calculation still needed for line 175
            num_output_components = (
                num_channels + (num_channels * (num_channels + 1)) // 2
            )  # Means, triangular A.
            if diagonal_covariance:
                num_output_components = num_channels * 2  # Means, diagonal of A.
            if debug:
                print("Num. output components:", num_output_components)
            # Call the NN with the current image etc.
            net_out, mask, img = self.models[Denoiser.MODEL](inp)
            if train:
                p = np.random.rand()
                if p <= 0.5:
                    inp_f = inp.flip(-1)
                else:
                    inp_f = inp.flip(-2)
                net_out_f, mask_f, img_f = self.models[Denoiser.MODEL](inp_f)
            if debug:
                print("Net output shape:", net_out.shape)
            mu_x = net_out[:, 0:num_channels, ...]  # Means (NCHW).
            A_c = net_out[
                :, num_channels:num_output_components, ...
            ]  # Components of triangular A.

            if debug:
                print("Shape of A_c:", A_c.shape)
            if num_channels == 1:
                sigma_x = A_c ** 2  # N1HW
            # elif num_channels == 3:
            #     if debug:
            #         print("Shape before permute:", A_c.shape)
            #     A_c = A_c.permute(0, 2, 3, 1)  # NHWC
            #     if debug:
            #         print("Shape after permute:", A_c.shape)
            #     if diagonal_covariance:
            #         c00 = A_c[..., 0] ** 2
            #         c11 = A_c[..., 1] ** 2
            #         c22 = A_c[..., 2] ** 2
            #         zro = torch.zeros(c00.shape())
            #         c0 = torch.stack([c00, zro, zro], dim=-1)  # NHW3
            #         c1 = torch.stack([zro, c11, zro], dim=-1)  # NHW3
            #         c2 = torch.stack([zro, zro, c22], dim=-1)  # NHW3
            #     else:
            #         # Calculate A^T * A
            #         c00 = A_c[..., 0] ** 2 + A_c[..., 1] ** 2 + A_c[..., 2] ** 2  # NHW
            #         c01 = A_c[..., 1] * A_c[..., 3] + A_c[..., 2] * A_c[..., 4]
            #         c02 = A_c[..., 2] * A_c[..., 5]
            #         c11 = A_c[..., 3] ** 2 + A_c[..., 4] ** 2
            #         c12 = A_c[..., 4] * A_c[..., 5]
            #         c22 = A_c[..., 5] ** 2
            #         c0 = torch.stack([c00, c01, c02], dim=-1)  # NHW3
            #         c1 = torch.stack([c01, c11, c12], dim=-1)  # NHW3
            #         c2 = torch.stack([c02, c12, c22], dim=-1)  # NHW3
            #     sigma_x = torch.stack([c0, c1, c2], dim=-1)  # NHW33

            # Data on which noise parameter estimation is based.
            if noise_params == NoiseValue.UNKNOWN_CONSTANT:
                # Global constant over the entire dataset.
                noise_est_out = self.l_params[Denoiser.ESTIMATED_SIGMA]
                # print('noise est out')
                # print(noise_est_out)
            elif noise_params == NoiseValue.UNKNOWN_VARIABLE:
                # Separate analysis network.
                param_est_net_out = self.models[Denoiser.SIGMA_ESTIMATOR](noisy_in)
                # print('param_est_net_out')
                # print(param_est_net_out)
                # print(param_est_net_out.shape)
                param_est_net_out = torch.mean(param_est_net_out, dim=(2, 3), keepdim=True)
                # print(param_est_net_out.shape)
                noise_est_out = param_est_net_out  # .type(torch.float64)

            # Cast remaining data into float64.
            # noisy_in = noisy_in.type(torch.float64)
            # noise_params_in = noise_params_in.type(torch.float64)

            # Remap noise estimate to ensure it is always positive and starts near zero.
            if noise_params != NoiseValue.KNOWN:
                # default pytorch vals: beta=1, threshold=20
                softplus = torch.nn.Softplus()  # yes this line is necessary, don't ask
                noise_est_out = softplus(noise_est_out - 4.0) + 1e-3

            # Distill noise parameters from learned/known data.
            if noise_style.startswith("gauss"):
                if noise_params == NoiseValue.KNOWN:
                    noise_std = torch.max(
                        noise_params_in, torch.tensor(1e-3)  # , dtype=torch.float64)
                    )  # N111
                else:
                    noise_std = noise_est_out
            elif noise_style.startswith(
                "poisson"
            ):  # Simple signal-dependent Poisson approximation [Hasinoff 2012].
                if noise_params == NoiseValue.KNOWN:
                    noise_std = (
                        torch.maximum(mu_x, torch.tensor(1e-3))  # , dtype=torch.float64))
                        / noise_params_in
                    ) ** 0.5  # NCHW
                else:
                    noise_std = (
                        torch.maximum(mu_x, torch.tensor(1e-3))  # , dtype=torch.float64))
                        * noise_est_out
                    ) ** 0.5  # NCHW

            # Casts and vars.
            # noise_std = noise_std.type(torch.float64)
            noise_std = noise_std.to(self.device)
            # I = tf.eye(num_channels, batch_shape=[1, 1, 1], dtype=tf.float64)
            I = torch.eye(num_channels, device=self.device)  # dtype=torch.float64
            I = I.reshape(
                1, 1, 1, num_channels, num_channels
            )  # Creates the same shape as the tensorflow thing did, wouldn't work for other batch shapes
            Ieps = I * 1e-6
            zero64 = torch.tensor(0.0, device=self.device)  # , dtype=torch.float64

            # Helpers.
            def batch_mvmul(m, v):  # Batched (M * v).
                return torch.sum(m * v[..., None, :], dim=-1)

            def batch_vtmv(v, m):  # Batched (v^T * M * v).
                return torch.sum(v[..., :, None] * v[..., None, :] * m, dim=[-2, -1])

            def batch_vvt(v):  # Batched (v * v^T).
                return v[..., :, None] * v[..., None, :]

            # Negative log-likelihood loss and posterior mean estimation.
            if noise_style.startswith("gauss") or noise_style.startswith("poisson"):
                if num_channels == 1:
                    sigma_n = noise_std ** 2  # N111 / N1HW
                    sigma_y = sigma_x + sigma_n  # N1HW. Total variance.
                    loss_out = ((noisy_in - mu_x) ** 2) / sigma_y + torch.log(
                        sigma_y
                    )  # N1HW
                    pme_out = (noisy_in * sigma_x + mu_x * sigma_n) / (
                        sigma_x + sigma_n
                    )  # N1HW
                    net_std_out = (sigma_x ** 0.5)[:, 0, ...]  # NHW
                    noise_std_out = noise_std[:, 0, ...]  # N11 / NHW
                    if noise_params != NoiseValue.KNOWN:
                        loss_out = loss_out - 0.05 * noise_std  # Balance regularization.
                    # print('loss_out')
                    # print(loss_out)
                    # print(loss_out.shape)
                else:
                    # Training loss.
                    noise_std_sqr = noise_std ** 2
                    sigma_n = (
                        noise_std_sqr.permute(0, 2, 3, 1)[..., None] * I
                    )  # NHWC1 * NHWCC = NHWCC
                    if debug:
                        print("sigma_n device:", sigma_n.device)
                    if debug:
                        print("sigma_x device:", sigma_x.device)
                    sigma_y = (
                        sigma_x + sigma_n
                    )  # NHWCC, total covariance matrix. Cannot be singular because sigma_n is at least a small diagonal.
                    if debug:
                        print("sigma_y device:", sigma_y.device)
                    sigma_y_inv = torch.inverse(sigma_y)  # NHWCC
                    mu_x2 = mu_x.permute(0, 2, 3, 1)  # NHWC
                    noisy_in2 = noisy_in.permute(0, 2, 3, 1)  # NHWC
                    diff = noisy_in2 - mu_x2  # NHWC
                    diff = -0.5 * batch_vtmv(diff, sigma_y_inv)  # NHW
                    dets = torch.det(sigma_y)  # NHW
                    dets = torch.max(
                        zero64, dets
                    )  # NHW. Avoid division by zero and negative square roots.
                    loss_out = 0.5 * torch.log(dets) - diff  # NHW
                    if noise_params != NoiseValue.KNOWN:
                        loss_out = loss_out - 0.1 * torch.mean(
                            noise_std, dim=1
                        )  # Balance regularization.

                    # Posterior mean estimate.
                    sigma_x_inv = torch.inverse(sigma_x + Ieps)  # NHWCC
                    sigma_n_inv = torch.inverse(sigma_n + Ieps)  # NHWCC
                    pme_c1 = torch.inverse(sigma_x_inv + sigma_n_inv + Ieps)  # NHWCC
                    pme_c2 = batch_mvmul(sigma_x_inv, mu_x2)  # NHWCC * NHWC -> NHWC
                    pme_c2 = pme_c2 + batch_mvmul(sigma_n_inv, noisy_in2)  # NHWC
                    pme_out = batch_mvmul(pme_c1, pme_c2)  # NHWC
                    pme_out = pme_out.permute(0, 3, 1, 2)  # NCHW

                    # Summary statistics.
                    net_std_out = torch.max(zero64, torch.det(sigma_x)) ** (
                        1.0 / 6.0
                    )  # NHW
                    noise_std_out = torch.max(zero64, torch.det(sigma_n)) ** (
                        1.0 / 6.0
                    )  # N11 / NHW

        
            loss_out = loss_out.view(loss_out.shape[0], -1).mean(1, keepdim=True)
            if train:
                # print('')
                mask_min = torch.min(mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2]*mask.shape[3]), dim=2)[0]
                mask_max = torch.max(mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2]*mask.shape[3]), dim=2)[0]
                rescaled_mask = (mask - mask_min[:,:,None,None])/(mask_max[:,:,None,None] - mask_min[:,:,None,None])
                # rescaled_mask = mask

                mask_min_f = torch.min(mask_f.reshape(mask.shape[0],mask.shape[1],mask.shape[2]*mask.shape[3]), dim=2)[0]
                mask_max_f = torch.max(mask_f.reshape(mask.shape[0],mask.shape[1],mask.shape[2]*mask.shape[3]), dim=2)[0]
                rescaled_mask_f = (mask_f - mask_min_f[:,:,None,None])/(mask_max_f[:,:,None,None] - mask_min_f[:,:,None,None])
                # rescaled_mask = nn.Sigmoid()(mask)
                # rescaled_mask_f = nn.Sigmoid()(mask_f)
                # rescaled_mask_f = mask_f
                m = nn.Sigmoid()
                # rescaled_mask = m(mask)
                sig_mask = m(7*(rescaled_mask-0.5))
                sig_mask_f = m(7*(rescaled_mask_f-0.5))
                if p <= 0.5:
                    sig_mask_f = sig_mask_f.flip(-1)
                if p > 0.5:
                    sig_mask_f = sig_mask_f.flip(-2)
                # print('target,',target[0,0,25:38,25:38])
                # mse_loss = nn.MSELoss()
                # diff = mse_loss(mask, target).mean()
                # print('mu_x', mu_x[0])
                # print('target,',target)
                fg = img * sig_mask
                bg = img * (1-sig_mask_f)
                fg_f = img*sig_mask_f
                bg_f = img*(1-sig_mask)
                bg_mean = torch.mean(bg, dim=(2,3))
                # print('bg_mean shape', bg_mean.shape, bg_mean[0])
                fg_mean = torch.mean(fg, dim=(2,3))
                # print('fg mean shape', fg_mean.shape, fg_mean[0])
                # bg_all = torch.ones(bg.size()).to(self.device)*bg_mean[:,:,None,None]
                
                pure_noise = torch.randn_like(mu_x)*bg_mean[:,:,None,None]
                pure_noise_mask = pure_noise * mask
                pure_contam = torch.zeros_like(mu_x)+torch.rand(1, device = self.device)
                one_lb = torch.ones(mu_x.shape[0], 1).view(-1).to(self.device)
                zero_lb = torch.zeros(mu_x.shape[0],1).view(-1).to(self.device)
                # pure_noise = noise_std * pure_noise
                # img_pred = self.models[Denoiser.PROB_ESTIMATOR](img).view(-1)
                # fg_pred = self.models[Denoiser.PROB_ESTIMATOR](fg).view(-1)
                # bg_pred = self.models[Denoiser.PROB_ESTIMATOR](bg).view(-1)
                # noise_pred = self.models[Denoiser.PROB_ESTIMATOR](pure_noise).view(-1)
                # fg_noise = fg 
                fg_noise = fg + bg_f
                bg_noise = bg + fg_f
                # print('fg,', mu_x)
                # print('img_pred,', img_pred)
                # print('fg_pred,',fg_pred)
                # print('noise_pred', noise_pred)
                # print('fg stats,', torch.min(fg), torch.max(fg))
                # print('noise stats,', torch.min(pure_noise), torch.max(pure_noise))
                noise_fg_pred = self.models[Denoiser.PROB_ESTIMATOR](fg_noise).view(-1)
                noise_bg_pred = self.models[Denoiser.PROB_ESTIMATOR](bg_noise).view(-1)
                noise_bgg_pred = self.models[Denoiser.PROB_ESTIMATOR](bg).view(-1)
                fg_f_pred = self.models[Denoiser.PROB_ESTIMATOR](img).view(-1)
                bg_f_pred = self.models[Denoiser.PROB_ESTIMATOR](img_f).view(-1)
                mse = nn.MSELoss()
                fg_bg_loss = mse(fg_f, fg) + mse(bg_f, bg)
                mask_loss = mse(sig_mask, sig_mask_f)
                pred_loss = mse(noise_fg_pred, fg_f_pred) + mse(noise_bg_pred, bg_f_pred)

                # criteria = nn.BCEWithLogitsLoss(reduction='mean')
                # loss_fg = criteria(fg_pred, one_lb)
                loss_img = torch.nn.ReLU()(1.0-fg_f_pred).mean()
                loss_bg = torch.nn.ReLU()(1.0+noise_bgg_pred).mean()
                # loss_fg = torch.nn.ReLU()(1.0-fg_pred).mean()
                # loss_bg = torch.nn.ReLU()(1.0+bg_pred).mean()
                # loss_noise = torch.nn.ReLU()(1.0+noise_pred).mean()
                # loss_fgn = torch.nn.ReLU()(1.0-noise_fg_pred).mean()
                loss_fgn_1 = (-1 - noise_fg_pred)
                loss_fgn_2 = (-1 - noise_bg_pred)
                loss_fgn = ((loss_fgn_1 < 0).float() * loss_fgn_1).mean() + ((loss_fgn_2 < 0).float() * loss_fgn_2).mean()

                loss_bgn_1 = (-1 + fg_f_pred)
                loss_bgn_2 = (-1 + bg_f_pred)
                loss_bgn = ((loss_bgn_1 < 0).float() * loss_bgn_1).mean() + ((loss_bgn_2 < 0).float() * loss_bgn_2).mean()

                # loss_fgn = torch.nn.ReLU()(1.0 - fg_f_pred)+torch.nn.ReLU()(1.0 - bg_f_pred)
                # loss_bgn = torch.nn.ReLU()(1.0+noise_bg_pred).mean()
                # loss_bgn = torch.nn.ReLU()(1.0 + noise_fg_pred) + torch.nn.ReLU()(1.0 + noise_bg_pred)
                pme_highlight = pme_out * mask
                # print('mask,', mask[0,0,20:40,20:40])
                # print('boundary mask', mask[0,0,0:10,0:10])
                # print('mask shape', mask.shape)
                # loss_bg = criteria(bg_pred, zero_lb)
                # loss_noise = criteria(noise_pred, zero_lb)
                # loss_fgn = criteria(noise_fg_pred, one_lb)
                # loss_bgn = criteria(noise_bg_pred, zero_lb)
                # print('mask center',mask[0,:,30:34,30:34].mean())
                # print('mask boundary',mask[0,:,0:2,:].mean())
                # loss_pred = loss_noise + loss_img + loss_fgn + loss_bgn
                loss_pred = -loss_fgn - loss_bgn + loss_bg + loss_img
                # loss_pred = 0
                # loss_pred = loss_fgn + loss_bg + loss_noise
                final_loss = alpha*loss_out + (1-alpha)* loss_pred + 0.5* (mask_loss + pred_loss)
            else:
                # _, mask, img = self.models[Denoiser.MODEL](inp[:,:,100-32:132,278-32:278+32])
                m = nn.Sigmoid()
                mask = mask[:,:,3:-3,3:-3]
                # mask_min = torch.quantile(mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2]*mask.shape[3]), 0.01 ,dim=2)
                mask_min = torch.min(mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2]*mask.shape[3]), dim = 2)[0]
                mask_max = torch.quantile(mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2]*mask.shape[3]), 0.995, dim=2)
                # mask_max = torch.max(mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2]*mask.shape[3]), dim = 2)[0]
                rescaled_mask = (mask - mask_min[:,:,None,None])/(mask_max[:,:,None,None] - mask_min[:,:,None,None])
                # rescaled_mask = m(mask)
                # rescaled_mask = mask
                # rescaled_mask = 
                # sig_mask = m(8*(rescaled_mask-0.5))
                # rescaled_mask = m(mask)
                sig_mask = m(7*(rescaled_mask-0.5))
                # print('target,',target[0,0,25:38,25:38])
                # mse_loss = nn.MSELoss()
                # diff = mse_loss(mask, target).mean()
                # print('mu_x', mu_x[0])
                # print('target,',target)
                # med_filt = medfilt(sig_mask, kernel_size = 5)
                fg = img[:,:,3:-3,3:-3] * sig_mask
                bg = img[:,:,3:-3,3:-3] * (1-sig_mask)

                bg_mean = torch.mean(bg, dim=(2,3))
                # pure_noise = torch.randn_like(mu_x)*bg_mean[:,:,None,None]
                
                # print('bg_mean shape', bg_mean.shape, bg_mean[0])
                fg_mean = torch.mean(fg, dim=(2,3))
                # print('fg mean shape', fg_mean.shape, fg_mean[0])
                bg_all = torch.ones(bg.size()).to(self.device)*bg_mean[:,:,None,None]
                bg_add = bg_all + bg
                # pure_noise_mask = pure_noise * mask
                # pure_noise = noise_std * pure_noise
                # pme_highlight = pme_out * mask
                # print('mask', mask)
                # mask = 1-mask
                fg_noise = fg 
                # bg_noise = mu_x[:,:,100-32:132,278-32:278+32]
                bg_noise = bg
                # pred_out = self.models[Denoiser.PROB_ESTIMATOR](pme_highlight)
                # mask = pred_out
                loss_pred = 0
                final_loss = loss_out
                mask_loss = 0
                pred_loss = 0


            return {
                PipelineOutput.INPUTS: data,
                PipelineOutput.IMG_MU: mu_x,
                PipelineOutput.TARGET: sig_mask,
                PipelineOutput.AUG_LOSS: mask_loss+pred_loss,
                # PipelineOutput.IMG_PME: pme_out,
                PipelineOutput.LOSS: final_loss,
                PipelineOutput.IMG_DENOISED: pme_out,
                PipelineOutput.DETECT_LOSS: loss_pred,
                PipelineOutput.DENOISE_LOSS: loss_out,
                PipelineOutput.NOISE_STD_DEV: noise_std_out,
                PipelineOutput.MODEL_STD_DEV: fg_noise,
                PipelineOutput.DETECT: rescaled_mask,
                PipelineOutput.GT: gt,
            }



    def _joint_pipeline(self, data: List, alpha: float, pi: float,**kwargs) -> Dict:
        debug = False
        noise_style = "gauss"
        noise_params = self.cfg[ConfigValue.NOISE_VALUE]

        # noise_style = self.cfg[ConfigValue.NOISE_STYLE]
        noise_style = "gauss"
        noise_params = self.cfg[ConfigValue.NOISE_VALUE]
        # print(len(data))
        if len(data) > 2:
            inp, target = data[DetectionDataset.INPUT], data[DetectionDataset.TARGET]
            metadata = data[DetectionDataset.METADATA]
            gt = metadata[DetectionDataset.Metadata.GT]
            # inps = torch.cat((inp, inp_aug), 0)
            # print('inp_aug', inp_aug)
            # print('concatenated')
            # print(inp.shape)
            ind = metadata[DetectionDataset.Metadata.INDEXES]
            # print('ind', ind)
            # print(inps.shape)
            target = target.to(self.device)
            # print('target',target)
            inp = inp.to(self.device)
            # inp_aug = inp_aug.to(self.device)
            noisy_in = inp
            # noisy_in_aug = inp_aug
            # print(inp[0])
            # print(inp_aug[0])
            input_shape = metadata[DetectionDataset.Metadata.IMAGE_SHAPE]
            num_channels = self.cfg[ConfigValue.IMAGE_CHANNELS]
            assert num_channels in [1, 3]

            diagonal_covariance = self.cfg[ConfigValue.DIAGONAL_COVARIANCE]
            if debug:
                print("Image shape:", input_shape)
            if debug:
                print("Num. channels:", num_channels)

            # Clean data distribution.
            # Calculation still needed for line 175
            num_output_components = (
                num_channels + (num_channels * (num_channels + 1)) // 2
            )  # Means, triangular A.
            if diagonal_covariance:
                num_output_components = num_channels * 2  # Means, diagonal of A.
            if debug:
                print("Num. output components:", num_output_components)
            # Call the NN with the current image etc.
            net_out = self.models[Denoiser.MODEL](inp)
            # det_aug, net_aug_out = self.models[Denoiser.MODEL](inp_aug)
            criteria_sup = nn.BCEWithLogitsLoss(reduction='mean')
            # criteria_cons = nn.MSELoss(reduction='mean')
            # criteria_kl = nn.KLDivLoss(log_target=True)
            

            # det_orig = det[:det.shape[0]//2]
            # det_aug = det[det.shape[0]//2:]
            # print('det', det)
            # print('det_orig', det_orig_sig)
            # print('det_aug', det_aug_sig)
            # det_orig_f = det_orig.view(det_orig.shape[0],-1)
            predict = net_out[:, -1,...]


            # det_aug_f = det_aug.view(det_aug.shape[0], -1)
            sig = nn.Sigmoid()
            predict_prob = sig(predict)
            pred_loss = modified_pu_loss(criteria_sup, pi, predict_prob, target)
            # softmax = nn.Softmax(dim=0)
            # det_orig_sig = sig(det_orig_f)
            # det_aug_sig = sig(det_aug_f)
            # det_orig_soft = torch.log(softmax(det_orig_f)+0.00001)
            # # print(det_orig_soft)

            # det_aug_soft = torch.log(softmax(det_aug_f)+0.00001)
            # print(det_aug_soft)
            # if target.shape[-1] > 1:
                # target = target.view(target.shape[0],-1)
            # print('2 target', target[target == 2])
            # print('target',target)
            # print('target', target.shape)
            # det_orig_f_zero = det_orig_f[target == 0]
            # det_orig_f_one = det_orig_f[target == 1]
            # print(det_orig_f_zero.shape[0])
            # print(det_orig_f_zero.shape[0]>0)
            # PU_Loss = PuLoss()
            # supervised_loss = PU_Loss(criteria_sup, pi, det_orig_f, target)
            # supervised_loss_aug = PU_Loss(criteria_sup, pi, det_aug_f, target)
            # if det_orig_f_one.shape[0] > 0 and det_orig_f_zero.shape[0] > 0:

            #     supervised_loss = 0.6* criteria_sup(det_orig_f_zero, target[target == 0]) + 0.4 * criteria_sup(det_orig_f_one, target[target == 1])
            # elif det_orig_f_one.shape[0] > 0 and det_orig_f_zero.shape[0] == 0:
            #     supervised_loss = criteria_sup(det_orig_f_one, target[target == 1])
            # elif det_orig_f_one.shape[0] == 0 and det_orig_f_zero.shape[0] > 0:
            #     supervised_loss = criteria_sup(det_orig_f_zero, target[target == 0])
            # else:
            #     supervised_loss = 0
            # print('supervised_loss', supervised_loss)
            # consistency_loss = js_div_loss_2d(det_orig, det_aug)
            # consistency_loss = criteria_cons(det_orig_sig, det_aug_sig)
            # consistency_kl = criteria_kl(det_orig_soft, det_aug_soft)
            # det_orig_sig_ones = det_orig_sig[target == 1]
            # det_orig_sig_zeros = det_orig_sig[target == 0]
            # det_aug_sig_ones = det_aug_sig[target == 1]
            # det_aug_sig_zeros = det_aug_sig[target == 0]
            # # if det_orig_sig_ones.shape[0] > 0:
            # #     consistency_loss = 0.8*criteria_cons(det_orig_sig_ones, det_aug_sig_ones) + 0.2*criteria_cons(det_orig_sig_zeros, det_aug_sig_zeros)
            # # else:
            # consistency_loss = criteria_cons(det_orig_sig, det_aug_sig)
            # print('consistency_loss', consistency_loss)
            # print('consistency_kl', consistency_kl)
            # print('supervised_loss', supervised_loss)
            # total_loss = (supervised_loss+0.1*consistency_loss)
            # print('supervised_loss loss', supervised_loss)
            # print('consistency_loss', consistency_loss)

            # total_loss = total_loss
            # print(total_loss)
            mu_x = net_out[:, 0:num_channels, ...]  # Means (NCHW).
            A_c = net_out[
                :, num_channels:num_output_components, ...
            ] 
            # det = det_orig
            # mu_x_aug = net_aug_out[:,0:num_channels,...]
            # A_c_aug = net_aug_out[:,num_channels:num_output_components,...]

        # print('A_c')
        # print(A_c.shape)
            if debug:
                print("Shape of A_c:", A_c.shape)
            if num_channels == 1:
                sigma_x = A_c ** 2  # N1HW
                # sigma_x_aug = A_c_aug ** 2
            elif num_channels == 3:
                if debug:
                    print("Shape before permute:", A_c.shape)
                A_c = A_c.permute(0, 2, 3, 1)  # NHWC
                if debug:
                    print("Shape after permute:", A_c.shape)
                if diagonal_covariance:
                    c00 = A_c[..., 0] ** 2
                    c11 = A_c[..., 1] ** 2
                    c22 = A_c[..., 2] ** 2
                    zro = torch.zeros(c00.shape())
                    c0 = torch.stack([c00, zro, zro], dim=-1)  # NHW3
                    c1 = torch.stack([zro, c11, zro], dim=-1)  # NHW3
                    c2 = torch.stack([zro, zro, c22], dim=-1)  # NHW3
                else:
                    # Calculate A^T * A
                    c00 = A_c[..., 0] ** 2 + A_c[..., 1] ** 2 + A_c[..., 2] ** 2  # NHW
                    c01 = A_c[..., 1] * A_c[..., 3] + A_c[..., 2] * A_c[..., 4]
                    c02 = A_c[..., 2] * A_c[..., 5]
                    c11 = A_c[..., 3] ** 2 + A_c[..., 4] ** 2
                    c12 = A_c[..., 4] * A_c[..., 5]
                    c22 = A_c[..., 5] ** 2
                    c0 = torch.stack([c00, c01, c02], dim=-1)  # NHW3
                    c1 = torch.stack([c01, c11, c12], dim=-1)  # NHW3
                    c2 = torch.stack([c02, c12, c22], dim=-1)  # NHW3
                sigma_x = torch.stack([c0, c1, c2], dim=-1)  # NHW33
            param_est_net_out = self.models[Denoiser.SIGMA_ESTIMATOR](inp)
            # param_est_net_out_aug = self.models[Denoiser.SIGMA_ESTIMATOR](inp_aug)
        
            param_est_net_out = torch.mean(param_est_net_out, dim=(2, 3), keepdim=True)
            # param_est_net_out_aug =  torch.mean(param_est_net_out_aug, dim=(2, 3), keepdim=True)
            # print('param_est_net_out_aug', param_est_net_out_aug)
            # print('param_est_net_out', param_est_net_out)
                # print(param_est_net_out.shape)
            noise_est_out = param_est_net_out
            # noise_est_out_aug = param_est_net_out_aug
            if noise_params != NoiseValue.KNOWN:
                # default pytorch vals: beta=1, threshold=20
                softplus = torch.nn.Softplus()  # yes this line is necessary, don't ask
                noise_est_out = softplus(noise_est_out - 4.0) + 1e-3
                # noise_est_out_aug = softplus(noise_est_out_aug - 4.0) + 1e-3
            noise_std = noise_est_out
            # noise_std_aug = noise_est_out_aug
            I = torch.eye(num_channels, device=self.device)  # dtype=torch.float64
            I = I.reshape(
                1, 1, 1, num_channels, num_channels
            )  # Creates the same shape as the tensorflow thing did, wouldn't work for other batch shapes
            Ieps = I * 1e-6
            zero64 = torch.tensor(0.0, device=self.device)  # , dtype=torch.float64

            # Helpers.
            def batch_mvmul(m, v):  # Batched (M * v).
                return torch.sum(m * v[..., None, :], dim=-1)

            def batch_vtmv(v, m):  # Batched (v^T * M * v).
                return torch.sum(v[..., :, None] * v[..., None, :] * m, dim=[-2, -1])

            def batch_vvt(v):  # Batched (v * v^T).
                return v[..., :, None] * v[..., None, :]

            # Negative log-likelihood loss and posterior mean estimation.
            if noise_style.startswith("gauss") or noise_style.startswith("poisson"):
                if num_channels == 1:
                    sigma_n = noise_std ** 2  # N111 / N1HW
                    # sigma_n_aug = noise_std_aug ** 2
                    sigma_y = sigma_x + sigma_n  # N1HW. Total variance.
                    # sigma_y_aug = sigma_x_aug + sigma_n_aug
                    loss_out = ((noisy_in - mu_x) ** 2) / sigma_y + torch.log(
                        sigma_y
                    )  # N1HW
                    # loss_out_aug = ((noisy_in_aug - mu_x_aug) ** 2) / sigma_y_aug + torch.log(sigma_y_aug)
                    pme_out = (noisy_in * sigma_x + mu_x * sigma_n) / (
                        sigma_x + sigma_n
                    )  # N1HW
                    net_std_out = (sigma_x ** 0.5)[:, 0, ...]  # NHW
                    noise_std_out = noise_std[:, 0, ...]  # N11 / NHW
                    if noise_params != NoiseValue.KNOWN:
                        loss_out = loss_out - 0.1 * noise_std  # Balance regularization.
                        # loss_out_aug = loss_out_aug - 0.05 * noise_std_aug
                    # print('loss_out')
                    # print(loss_out)
                    # print(loss_out.shape)
                else:
                    # Training loss.
                    noise_std_sqr = noise_std ** 2
                    sigma_n = (
                        noise_std_sqr.permute(0, 2, 3, 1)[..., None] * I
                    )  # NHWC1 * NHWCC = NHWCC
                    if debug:
                        print("sigma_n device:", sigma_n.device)
                    if debug:
                        print("sigma_x device:", sigma_x.device)
                    sigma_y = (
                        sigma_x + sigma_n
                    )  # NHWCC, total covariance matrix. Cannot be singular because sigma_n is at least a small diagonal.
                    if debug:
                        print("sigma_y device:", sigma_y.device)
                    sigma_y_inv = torch.inverse(sigma_y)  # NHWCC
                    mu_x2 = mu_x.permute(0, 2, 3, 1)  # NHWC
                    noisy_in2 = noisy_in.permute(0, 2, 3, 1)  # NHWC
                    diff = noisy_in2 - mu_x2  # NHWC
                    diff = -0.5 * batch_vtmv(diff, sigma_y_inv)  # NHW
                    dets = torch.det(sigma_y)  # NHW
                    dets = torch.max(
                        zero64, dets
                    )  # NHW. Avoid division by zero and negative square roots.
                    loss_out = 0.5 * torch.log(dets) - diff  # NHW
                    if noise_params != NoiseValue.KNOWN:
                        loss_out = loss_out - 0.1 * torch.mean(
                            noise_std, dim=1
                        )  # Balance regularization.

                    # Posterior mean estimate.
                    sigma_x_inv = torch.inverse(sigma_x + Ieps)  # NHWCC
                    sigma_n_inv = torch.inverse(sigma_n + Ieps)  # NHWCC
                    pme_c1 = torch.inverse(sigma_x_inv + sigma_n_inv + Ieps)  # NHWCC
                    pme_c2 = batch_mvmul(sigma_x_inv, mu_x2)  # NHWCC * NHWC -> NHWC
                    pme_c2 = pme_c2 + batch_mvmul(sigma_n_inv, noisy_in2)  # NHWC
                    pme_out = batch_mvmul(pme_c1, pme_c2)  # NHWC
                    pme_out = pme_out.permute(0, 3, 1, 2)  # NCHW

                    # Summary statistics.
                    net_std_out = torch.max(zero64, torch.det(sigma_x)) ** (
                        1.0 / 6.0
                    )  # NHW
                    noise_std_out = torch.max(zero64, torch.det(sigma_n)) ** (
                        1.0 / 6.0
                    )  # N11 / NHW

            # mu_x = mean of x
            # pme_out = posterior mean estimate
            # loss_out = loss
            # net_std_out = std estimate from nn
            # noise_std_out = predicted noise std?
            # return mu_x, pme_out, loss_out, net_std_out, noise_std_out
            loss_out = loss_out.view(loss_out.shape[0], -1).mean(1, keepdim=True)
            # loss_out_aug = loss_out_aug.view(loss_out_aug.shape[0],-1).mean(1, keepdim=True)
            # print('loss_out_aug', loss_out_aug)
            # aug_loss = nn.MSELoss(reduction="none")(noise_est_out, noise_est_out_aug)
            # print('aug_loss',aug_loss)
            # print('loss_out', loss_out)
            final_loss = alpha*loss_out + (1-alpha)* pred_loss
            predict_prob = predict_prob.unsqueeze(1)
            # print('inp', inp.shape)
            # print('predict_prob',predict_prob.shape)
            # print('mu', mu_x.shape)
            # print('total_loss', total_loss)
            # print('loss_out', loss_out)
            # print('final_loss', final_loss.shape)
            # print('final_loss', final_loss)
            # det = 
            return {
                PipelineOutput.INPUTS: data,
                PipelineOutput.TARGET: target,
                PipelineOutput.IMG_MU: mu_x,
                PipelineOutput.IMG_DENOISED: pme_out,
                PipelineOutput.LOSS: final_loss,
                PipelineOutput.DETECT_LOSS: pred_loss,
                PipelineOutput.DENOISE_LOSS: loss_out,
                PipelineOutput.NOISE_STD_DEV: noise_std_out,
                PipelineOutput.MODEL_STD_DEV: net_std_out,
                PipelineOutput.DETECT: predict_prob,
                PipelineOutput.GT: gt,
            }
        else:
            inp, target = data[DetectionDataset.INPUT], data[DetectionDataset.TARGET]
            noisy_in = inp.to(self.device)
            target = target.to(self.device)
            metadata = data[DetectionDataset.METADATA]
            gt = metadata[DetectionDataset.Metadata.GT]
            # print('gt', gt.shape)
            # print('noisy_in',noisy_in.shape)
            # print('target', target.shape)
            #noise_params_in = metadata[NoisyDataset.Metadata.INPUT_NOISE_VALUES]

            # config for noise params/style
            # noise_style = self.cfg[ConfigValue.NOISE_STYLE]
            noise_style = "gauss"
            noise_params = self.cfg[ConfigValue.NOISE_VALUE]

            # Equivalent of blindspot_pipeline
            input_shape = metadata[DetectionDataset.Metadata.IMAGE_SHAPE]
            num_channels = self.cfg[ConfigValue.IMAGE_CHANNELS]
            assert num_channels in [1, 3]

            diagonal_covariance = self.cfg[ConfigValue.DIAGONAL_COVARIANCE]
            if debug:
                print("Image shape:", input_shape)
            if debug:
                print("Num. channels:", num_channels)

            # Clean data distribution.
            # Calculation still needed for line 175
            num_output_components = (
                num_channels + (num_channels * (num_channels + 1)) // 2
            )  # Means, triangular A.
            if diagonal_covariance:
                num_output_components = num_channels * 2  # Means, diagonal of A.
            if debug:
                print("Num. output components:", num_output_components)
            # Call the NN with the current image etc.
            net_out = self.models[Denoiser.MODEL](noisy_in)
            if not all(torch.tensor(det.shape) == torch.tensor(target.shape)):
                print('not equal')
                
            criteria = nn.BCEWithLogitsLoss(reduction="none")
            det_f = det.view(det.shape[0],-1)
            # print('det')
            # print(det.shape)
            # print('target')
            # print(target.shape)
            # print(det_f)
            # print('target', torch.max(target))
            if target.shape[-1] > 1:
                target = target.view(target.shape[0],-1)
            total_loss = criteria(det_f, target)
            # print('det loss')
            # print(det_loss)
            # print(det_loss.shape)
            # print('model')
            # print(self.models[Denoiser.MODEL])
            # net_out = net_out.type(torch.float64)
            # print('net out')
            # print(net_out.shape)

            if debug:
                print("Net output shape:", net_out.shape)
            mu_x = net_out[:, 0:num_channels, ...]  # Means (NCHW).
            A_c = net_out[
                :, num_channels:num_output_components, ...
            ] 
            if debug:
                print("Shape of A_c:", A_c.shape)
            if num_channels == 1:
                sigma_x = A_c ** 2  # N1HW
            elif num_channels == 3:
                if debug:
                    print("Shape before permute:", A_c.shape)
                A_c = A_c.permute(0, 2, 3, 1)  # NHWC
                if debug:
                    print("Shape after permute:", A_c.shape)
                if diagonal_covariance:
                    c00 = A_c[..., 0] ** 2
                    c11 = A_c[..., 1] ** 2
                    c22 = A_c[..., 2] ** 2
                    zro = torch.zeros(c00.shape())
                    c0 = torch.stack([c00, zro, zro], dim=-1)  # NHW3
                    c1 = torch.stack([zro, c11, zro], dim=-1)  # NHW3
                    c2 = torch.stack([zro, zro, c22], dim=-1)  # NHW3
                else:
                    # Calculate A^T * A
                    c00 = A_c[..., 0] ** 2 + A_c[..., 1] ** 2 + A_c[..., 2] ** 2  # NHW
                    c01 = A_c[..., 1] * A_c[..., 3] + A_c[..., 2] * A_c[..., 4]
                    c02 = A_c[..., 2] * A_c[..., 5]
                    c11 = A_c[..., 3] ** 2 + A_c[..., 4] ** 2
                    c12 = A_c[..., 4] * A_c[..., 5]
                    c22 = A_c[..., 5] ** 2
                    c0 = torch.stack([c00, c01, c02], dim=-1)  # NHW3
                    c1 = torch.stack([c01, c11, c12], dim=-1)  # NHW3
                    c2 = torch.stack([c02, c12, c22], dim=-1)  # NHW3
                sigma_x = torch.stack([c0, c1, c2], dim=-1)  # NHW33
            param_est_net_out = self.models[Denoiser.SIGMA_ESTIMATOR](noisy_in)
        
            param_est_net_out = torch.mean(param_est_net_out, dim=(2, 3), keepdim=True)
                # print(param_est_net_out.shape)
            noise_est_out = param_est_net_out
            if noise_params != NoiseValue.KNOWN:
                # default pytorch vals: beta=1, threshold=20
                softplus = torch.nn.Softplus()  # yes this line is necessary, don't ask
                noise_est_out = softplus(noise_est_out - 4.0) + 1e-3
            noise_std = noise_est_out
            I = torch.eye(num_channels, device=self.device)  # dtype=torch.float64
            I = I.reshape(
                1, 1, 1, num_channels, num_channels
            )  # Creates the same shape as the tensorflow thing did, wouldn't work for other batch shapes
            Ieps = I * 1e-6
            zero64 = torch.tensor(0.0, device=self.device)  # , dtype=torch.float64

            # Helpers.
            def batch_mvmul(m, v):  # Batched (M * v).
                return torch.sum(m * v[..., None, :], dim=-1)

            def batch_vtmv(v, m):  # Batched (v^T * M * v).
                return torch.sum(v[..., :, None] * v[..., None, :] * m, dim=[-2, -1])

            def batch_vvt(v):  # Batched (v * v^T).
                return v[..., :, None] * v[..., None, :]

            # Negative log-likelihood loss and posterior mean estimation.
            if noise_style.startswith("gauss") or noise_style.startswith("poisson"):
                if num_channels == 1:
                    sigma_n = noise_std ** 2  # N111 / N1HW
                    sigma_y = sigma_x + sigma_n  # N1HW. Total variance.
                    loss_out = ((noisy_in - mu_x) ** 2) / sigma_y + torch.log(
                        sigma_y
                    )  # N1HW
                    pme_out = (noisy_in * sigma_x + mu_x * sigma_n) / (
                        sigma_x + sigma_n
                    )  # N1HW
                    net_std_out = (sigma_x ** 0.5)[:, 0, ...]  # NHW
                    noise_std_out = noise_std[:, 0, ...]  # N11 / NHW
                    if noise_params != NoiseValue.KNOWN:
                        loss_out = loss_out - 0.1 * noise_std  # Balance regularization.
                    # print('loss_out')
                    # print(loss_out)
                    # print(loss_out.shape)
                else:
                    # Training loss.
                    noise_std_sqr = noise_std ** 2
                    sigma_n = (
                        noise_std_sqr.permute(0, 2, 3, 1)[..., None] * I
                    )  # NHWC1 * NHWCC = NHWCC
                    if debug:
                        print("sigma_n device:", sigma_n.device)
                    if debug:
                        print("sigma_x device:", sigma_x.device)
                    sigma_y = (
                        sigma_x + sigma_n
                    )  # NHWCC, total covariance matrix. Cannot be singular because sigma_n is at least a small diagonal.
                    if debug:
                        print("sigma_y device:", sigma_y.device)
                    sigma_y_inv = torch.inverse(sigma_y)  # NHWCC
                    mu_x2 = mu_x.permute(0, 2, 3, 1)  # NHWC
                    noisy_in2 = noisy_in.permute(0, 2, 3, 1)  # NHWC
                    diff = noisy_in2 - mu_x2  # NHWC
                    diff = -0.5 * batch_vtmv(diff, sigma_y_inv)  # NHW
                    dets = torch.det(sigma_y)  # NHW
                    dets = torch.max(
                        zero64, dets
                    )  # NHW. Avoid division by zero and negative square roots.
                    loss_out = 0.5 * torch.log(dets) - diff  # NHW
                    if noise_params != NoiseValue.KNOWN:
                        loss_out = loss_out - 0.1 * torch.mean(
                            noise_std, dim=1
                        )  # Balance regularization.

                    # Posterior mean estimate.
                    sigma_x_inv = torch.inverse(sigma_x + Ieps)  # NHWCC
                    sigma_n_inv = torch.inverse(sigma_n + Ieps)  # NHWCC
                    pme_c1 = torch.inverse(sigma_x_inv + sigma_n_inv + Ieps)  # NHWCC
                    pme_c2 = batch_mvmul(sigma_x_inv, mu_x2)  # NHWCC * NHWC -> NHWC
                    pme_c2 = pme_c2 + batch_mvmul(sigma_n_inv, noisy_in2)  # NHWC
                    pme_out = batch_mvmul(pme_c1, pme_c2)  # NHWC
                    pme_out = pme_out.permute(0, 3, 1, 2)  # NCHW

                    # Summary statistics.
                    net_std_out = torch.max(zero64, torch.det(sigma_x)) ** (
                        1.0 / 6.0
                    )  # NHW
                    noise_std_out = torch.max(zero64, torch.det(sigma_n)) ** (
                        1.0 / 6.0
                    )  # N11 / NHW

            # mu_x = mean of x
            # pme_out = posterior mean estimate
            # loss_out = loss
            # net_std_out = std estimate from nn
            # noise_std_out = predicted noise std?
            # return mu_x, pme_out, loss_out, net_std_out, noise_std_out
            loss_out = loss_out.view(loss_out.shape[0], -1).mean(1, keepdim=True)
            final_loss = .5 * loss_out + 0.5 * total_loss

            return {
                PipelineOutput.INPUTS: data,
                PipelineOutput.TARGET: target,
                PipelineOutput.IMG_MU: mu_x,
                PipelineOutput.IMG_DENOISED: pme_out,
                PipelineOutput.GT: gt,
                PipelineOutput.LOSS: final_loss,
                PipelineOutput.NOISE_STD_DEV: noise_std_out,
                PipelineOutput.MODEL_STD_DEV: net_std_out,
                PipelineOutput.DETECT: det,
            }


        

    def _ssdn_pipeline(self, data: List, **kwargs) -> Dict:
        debug = False

        inp, target = data[DetectionDataset.INPUT], data[DetectionDataset.TARGET]
        # print('input')
        # print(inp.shape)
        noisy_in = inp.to(self.device)
        target = target.to(self.device)
        # noisy_params_in = standard deviation of noise
        metadata = data[DetectionDataset.METADATA]
        #noise_params_in = metadata[NoisyDataset.Metadata.INPUT_NOISE_VALUES]

        # config for noise params/style
        # noise_style = self.cfg[ConfigValue.NOISE_STYLE]
        noise_style = "gauss"
        noise_params = self.cfg[ConfigValue.NOISE_VALUE]

        # Equivalent of blindspot_pipeline
        input_shape = metadata[DetectionDataset.Metadata.IMAGE_SHAPE]
        num_channels = self.cfg[ConfigValue.IMAGE_CHANNELS]
        assert num_channels in [1, 3]

        diagonal_covariance = self.cfg[ConfigValue.DIAGONAL_COVARIANCE]

        if debug:
            print("Image shape:", input_shape)
        if debug:
            print("Num. channels:", num_channels)

        # Clean data distribution.
        # Calculation still needed for line 175
        num_output_components = (
            num_channels + (num_channels * (num_channels + 1)) // 2
        )  # Means, triangular A.
        if diagonal_covariance:
            num_output_components = num_channels * 2  # Means, diagonal of A.
        if debug:
            print("Num. output components:", num_output_components)
        # Call the NN with the current image etc.
        det, net_out = self.models[Denoiser.MODEL](noisy_in)
        # print('model')
        # print('det')
        # print(det.shape)
        # print(self.models[Denoiser.MODEL])
        # net_out = net_out.type(torch.float64)
        # print('net out')
        # print(net_out.shape)

        if debug:
            print("Net output shape:", net_out.shape)
        mu_x = net_out[:, 0:num_channels, ...]  # Means (NCHW).
        A_c = net_out[
            :, num_channels:num_output_components, ...
        ]  # Components of triangular A.
        # print('A_c')
        # print(A_c.shape)
        if debug:
            print("Shape of A_c:", A_c.shape)
        if num_channels == 1:
            sigma_x = A_c ** 2  # N1HW
        elif num_channels == 3:
            if debug:
                print("Shape before permute:", A_c.shape)
            A_c = A_c.permute(0, 2, 3, 1)  # NHWC
            if debug:
                print("Shape after permute:", A_c.shape)
            if diagonal_covariance:
                c00 = A_c[..., 0] ** 2
                c11 = A_c[..., 1] ** 2
                c22 = A_c[..., 2] ** 2
                zro = torch.zeros(c00.shape())
                c0 = torch.stack([c00, zro, zro], dim=-1)  # NHW3
                c1 = torch.stack([zro, c11, zro], dim=-1)  # NHW3
                c2 = torch.stack([zro, zro, c22], dim=-1)  # NHW3
            else:
                # Calculate A^T * A
                c00 = A_c[..., 0] ** 2 + A_c[..., 1] ** 2 + A_c[..., 2] ** 2  # NHW
                c01 = A_c[..., 1] * A_c[..., 3] + A_c[..., 2] * A_c[..., 4]
                c02 = A_c[..., 2] * A_c[..., 5]
                c11 = A_c[..., 3] ** 2 + A_c[..., 4] ** 2
                c12 = A_c[..., 4] * A_c[..., 5]
                c22 = A_c[..., 5] ** 2
                c0 = torch.stack([c00, c01, c02], dim=-1)  # NHW3
                c1 = torch.stack([c01, c11, c12], dim=-1)  # NHW3
                c2 = torch.stack([c02, c12, c22], dim=-1)  # NHW3
            sigma_x = torch.stack([c0, c1, c2], dim=-1)  # NHW33

        # Data on which noise parameter estimation is based.
        if noise_params == NoiseValue.UNKNOWN_CONSTANT:
            # Global constant over the entire dataset.
            noise_est_out = self.l_params[Denoiser.ESTIMATED_SIGMA]
            # print('noise est out')
            # print(noise_est_out)
        elif noise_params == NoiseValue.UNKNOWN_VARIABLE:
            # Separate analysis network.
            param_est_net_out = self.models[Denoiser.SIGMA_ESTIMATOR](noisy_in)
            # print('param_est_net_out')
            # print(param_est_net_out)
            # print(param_est_net_out.shape)
            param_est_net_out = torch.mean(param_est_net_out, dim=(2, 3), keepdim=True)
            # print(param_est_net_out.shape)
            noise_est_out = param_est_net_out  # .type(torch.float64)

        # Cast remaining data into float64.
        # noisy_in = noisy_in.type(torch.float64)
        # noise_params_in = noise_params_in.type(torch.float64)

        # Remap noise estimate to ensure it is always positive and starts near zero.
        if noise_params != NoiseValue.KNOWN:
            # default pytorch vals: beta=1, threshold=20
            softplus = torch.nn.Softplus()  # yes this line is necessary, don't ask
            noise_est_out = softplus(noise_est_out - 4.0) + 1e-3

        # Distill noise parameters from learned/known data.
        if noise_style.startswith("gauss"):
            if noise_params == NoiseValue.KNOWN:
                noise_std = torch.max(
                    noise_params_in, torch.tensor(1e-3)  # , dtype=torch.float64)
                )  # N111
            else:
                noise_std = noise_est_out
        elif noise_style.startswith(
            "poisson"
        ):  # Simple signal-dependent Poisson approximation [Hasinoff 2012].
            if noise_params == NoiseValue.KNOWN:
                noise_std = (
                    torch.maximum(mu_x, torch.tensor(1e-3))  # , dtype=torch.float64))
                    / noise_params_in
                ) ** 0.5  # NCHW
            else:
                noise_std = (
                    torch.maximum(mu_x, torch.tensor(1e-3))  # , dtype=torch.float64))
                    * noise_est_out
                ) ** 0.5  # NCHW

        # Casts and vars.
        # noise_std = noise_std.type(torch.float64)
        noise_std = noise_std.to(self.device)
        # I = tf.eye(num_channels, batch_shape=[1, 1, 1], dtype=tf.float64)
        I = torch.eye(num_channels, device=self.device)  # dtype=torch.float64
        I = I.reshape(
            1, 1, 1, num_channels, num_channels
        )  # Creates the same shape as the tensorflow thing did, wouldn't work for other batch shapes
        Ieps = I * 1e-6
        zero64 = torch.tensor(0.0, device=self.device)  # , dtype=torch.float64

        # Helpers.
        def batch_mvmul(m, v):  # Batched (M * v).
            return torch.sum(m * v[..., None, :], dim=-1)

        def batch_vtmv(v, m):  # Batched (v^T * M * v).
            return torch.sum(v[..., :, None] * v[..., None, :] * m, dim=[-2, -1])

        def batch_vvt(v):  # Batched (v * v^T).
            return v[..., :, None] * v[..., None, :]

        # Negative log-likelihood loss and posterior mean estimation.
        if noise_style.startswith("gauss") or noise_style.startswith("poisson"):
            if num_channels == 1:
                sigma_n = noise_std ** 2  # N111 / N1HW
                sigma_y = sigma_x + sigma_n  # N1HW. Total variance.
                loss_out = ((noisy_in - mu_x) ** 2) / sigma_y + torch.log(
                    sigma_y
                )  # N1HW
                pme_out = (noisy_in * sigma_x + mu_x * sigma_n) / (
                    sigma_x + sigma_n
                )  # N1HW
                net_std_out = (sigma_x ** 0.5)[:, 0, ...]  # NHW
                noise_std_out = noise_std[:, 0, ...]  # N11 / NHW
                if noise_params != NoiseValue.KNOWN:
                    loss_out = loss_out - 0.05 * noise_std  # Balance regularization.
                # print('loss_out')
                # print(loss_out)
                # print(loss_out.shape)
            else:
                # Training loss.
                noise_std_sqr = noise_std ** 2
                sigma_n = (
                    noise_std_sqr.permute(0, 2, 3, 1)[..., None] * I
                )  # NHWC1 * NHWCC = NHWCC
                if debug:
                    print("sigma_n device:", sigma_n.device)
                if debug:
                    print("sigma_x device:", sigma_x.device)
                sigma_y = (
                    sigma_x + sigma_n
                )  # NHWCC, total covariance matrix. Cannot be singular because sigma_n is at least a small diagonal.
                if debug:
                    print("sigma_y device:", sigma_y.device)
                sigma_y_inv = torch.inverse(sigma_y)  # NHWCC
                mu_x2 = mu_x.permute(0, 2, 3, 1)  # NHWC
                noisy_in2 = noisy_in.permute(0, 2, 3, 1)  # NHWC
                diff = noisy_in2 - mu_x2  # NHWC
                diff = -0.5 * batch_vtmv(diff, sigma_y_inv)  # NHW
                dets = torch.det(sigma_y)  # NHW
                dets = torch.max(
                    zero64, dets
                )  # NHW. Avoid division by zero and negative square roots.
                loss_out = 0.5 * torch.log(dets) - diff  # NHW
                if noise_params != NoiseValue.KNOWN:
                    loss_out = loss_out - 0.1 * torch.mean(
                        noise_std, dim=1
                    )  # Balance regularization.

                # Posterior mean estimate.
                sigma_x_inv = torch.inverse(sigma_x + Ieps)  # NHWCC
                sigma_n_inv = torch.inverse(sigma_n + Ieps)  # NHWCC
                pme_c1 = torch.inverse(sigma_x_inv + sigma_n_inv + Ieps)  # NHWCC
                pme_c2 = batch_mvmul(sigma_x_inv, mu_x2)  # NHWCC * NHWC -> NHWC
                pme_c2 = pme_c2 + batch_mvmul(sigma_n_inv, noisy_in2)  # NHWC
                pme_out = batch_mvmul(pme_c1, pme_c2)  # NHWC
                pme_out = pme_out.permute(0, 3, 1, 2)  # NCHW

                # Summary statistics.
                net_std_out = torch.max(zero64, torch.det(sigma_x)) ** (
                    1.0 / 6.0
                )  # NHW
                noise_std_out = torch.max(zero64, torch.det(sigma_n)) ** (
                    1.0 / 6.0
                )  # N11 / NHW

        # mu_x = mean of x
        # pme_out = posterior mean estimate
        # loss_out = loss
        # net_std_out = std estimate from nn
        # noise_std_out = predicted noise std?
        # return mu_x, pme_out, loss_out, net_std_out, noise_std_out
        loss_out = loss_out.view(loss_out.shape[0], -1).mean(1, keepdim=True)
        # print('loss')
        # unpad_mu = DetectionDataset.unpad(mu_x, target, metadata, None)[0]
        # print('mu_x', mu_x.shape)
        # print('unpad_mu',len(unpad_mu))
        # print(unpad_mu[0].shape)
        # print('final loss out')
        # print(loss_out.shape)
        return {
            PipelineOutput.INPUTS: data,
            PipelineOutput.IMG_MU: mu_x,
            PipelineOutput.TARGET: target,
            # PipelineOutput.IMG_PME: pme_out,
            PipelineOutput.IMG_DENOISED: pme_out,
            PipelineOutput.LOSS: loss_out,
            PipelineOutput.NOISE_STD_DEV: noise_std_out,
            PipelineOutput.MODEL_STD_DEV: net_std_out,
        }

    def state_dict(self, params_only: bool = False) -> Dict:
        state_dict = state_dict = super().state_dict()
        if not params_only:
            state_dict["cfg"] = self.cfg
        return state_dict

    @staticmethod
    def from_state_dict(state_dict: Dict, mode: str) -> Denoiser:
        denoiser = Denoiser(state_dict["cfg"], mode = mode)
        denoiser.load_state_dict(state_dict, strict=False)
        return denoiser

    # @staticmethod
    # def from_state_dict(state_dict: Dict) -> Denoiser:
    #     denoiser = Denoiser(state_dict["cfg"])
    #     denoiser.load_state_dict(state_dict, strict=False)
    #     return denoiser

    def config_name(self) -> str:
        return spr_pick.cfg.config_name(self.cfg)
