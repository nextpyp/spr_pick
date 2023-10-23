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
from spr_pick.models import JointNetwork, DualNetwork, DualNetworkShallow, DualNetworkShallower
from spr_pick.models import ResNet6, ResNet8, ResNet16
from spr_pick.models import LinearClassifier
from spr_pick.datasets import DetectionDataset
from spr_pick.models import NoiseEstNetwork

from typing import Dict, List
from spr_pick.utils.losses import js_div_loss_2d
from spr_pick.utils.losses import PuLoss, modified_pu_loss, PULoss, FocalLoss

def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

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
        print('using dual model ...')
        # self.add_model(
        #     Denoiser.MODEL, 
        #     DualNetwork(
        #         in_channels=in_channels,
        #         out_channels=2,
        #         blindspot=self.cfg[ConfigValue.BLINDSPOT],
        #         detect=True,
        #     ),
        # )
        self.add_model(
            Denoiser.MODEL, 
            JointNetwork(
                in_channels=in_channels,
                out_channels=2,
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
                DualNetworkShallow(
                    in_channels = in_channels,
                    out_channels = 1,
                    blindspot=False,
                    detect=False,
                ),
            )
            # self.add_model(
            #     Denoiser.PROB_ESTIMATOR,
            #     Detector(),
            #     )
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
        return self.models[Denoiser.MODEL].fill(stride=stride)

    def unfill(self):
        return self.models[Denoiser.MODEL].unfill()

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

    def run_pipeline(self, data: List, alpha = 0, tau = 0, train = True, **kwargs):
        if self.cfg[ConfigValue.PIPELINE] == Pipeline.MSE and self.mode == "denoise":
            return self._mse_pipeline(data, **kwargs)
        elif self.cfg[ConfigValue.PIPELINE] == Pipeline.SSDN and self.mode == "denoise":
            return self._ssdn_pipeline(data, **kwargs)
        elif self.cfg[ConfigValue.PIPELINE] == Pipeline.MASK_MSE and self.mode == "denoise":
            return self._mask_mse_pipeline(data, **kwargs)
        elif self.mode == "joint":
            return self._new_pipeline(data, alpha, tau, train = train, **kwargs)

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

    

    def _new_pipeline(self, data: List, alpha: float, tau: float, train: bool, **kwargs) -> Dict:
        debug = False
        # noise_style = "gauss"
        noise_style = self.cfg[ConfigValue.NOISE_STYLE]
        # print("noise_style", noise_style)
        noise_params = self.cfg[ConfigValue.NOISE_VALUE]
        # print('len_data', len(data))
        # test_data = None
        if len(data) > 2:
            #target should all be one, only sample labeled ones 
            inp, target, hm, hm_small = data[DetectionDataset.INPUT], data[DetectionDataset.TARGET], data[DetectionDataset.HM],data[DetectionDataset.HM_SMALL]
            metadata = data[DetectionDataset.METADATA]
            gt = metadata[DetectionDataset.Metadata.GT]
            ind = metadata[DetectionDataset.Metadata.INDEXES]
            target = target.to(self.device)
            # print('target', target.shape)
            # print(target)
            inp = inp.to(self.device)
            hm = hm.to(self.device)
            hm_small = hm_small.to(self.device)
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
            # if train:
            net_out, hm_p = self.models[Denoiser.MODEL](inp)
            # net_out, _ = self.models[Denoiser.MODEL](inp)
            # hm_p_p = hm_p.copy()
            # hm_p = net_out[:, num_output_components:3, ...]
            # print('hm_p', hm_p.shape)
            hm_p_p = hm_p
            hm_p = _sigmoid(hm_p)
            
            # hm_ps = _sigmoid(hm_ps)
            if train:
                # cr_loss = UnbiasedConLoss(0.07, alpha)
                p = np.random.rand()
                if p <= 0.5:
                    inp_f = inp.flip(-1)
                else:
                    inp_f = inp.flip(-2)
                net_out_f, hm_p_f  = self.models[Denoiser.MODEL](inp_f)
                # net_out_f, _ = self.models[Denoiser.MODEL](inp_f)

                # hm_p_f = net_out_f[:, num_output_components:3, ...]
                if p <= 0.5:
                    hm_p_f = hm_p_f.flip(-1)
                #     # output_fm_cr = output_fm_cr.flip(-1)
                else:
                    hm_p_f = hm_p_f.flip(-2)
                #     # output_fm_cr = output_fm_cr.flip(-2)
                # print('hm_p', hm_p.shape)
                hm_p_f = _sigmoid(hm_p_f)
                # criteria = FocalLoss()
                criteria = nn.BCELoss()
                pu_loss = PuLoss()
                # pu_loss = PULoss(tau = tau)
                # print('focal loss?')
                pred_loss = pu_loss(criteria, tau, hm_p, target) 
                # pred_loss = pu_loss(criteria,tau, hm_p, hm)
                # pred_loss = pu_loss(hm_p, hm)
                # pred_loss_s = pu_loss(hm_p, hm_small)
                # pred_loss = pred_loss + (debiased_loss_sup + 0.1*debiased_loss_unsup)
                # hm_p_f_flat = hm_p_f.reshape((1, -1))


            if debug:
                print("Net output shape:", net_out.shape)
            mu_x = net_out[:, 0:num_channels, ...]  # Means (NCHW).
            A_c = net_out[
                :, num_channels:num_output_components, ...
            ]  # Components of triangular A.

            
            
            mse_loss = nn.MSELoss()
            
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
                    net_std_out = net_std_out.unsqueeze(0)
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
                consis_loss = mse_loss(hm_p, hm_p_f)
                # print('alpha', alpha)
                # final_loss = alpha*loss_out + (1-alpha)* pred_loss_s + 0.1* (consis_loss)
                final_loss = alpha*loss_out + (1-alpha)*pred_loss + 0.1 * consis_loss
            else:
                # hm_p = nn.MaxPool2d(3, stride=1, padding=1)(hm_p)
                # print('testttttt')
                # _, mask, img = self.models[Denoiser.MODEL](inp[:,:,100-32:132,278-32:278+32])
                # m = nn.Sigmoid()
                # hm_p = _sigmoid()
                # mask = mask[:,:,3:-3,3:-3]
                # mask = hm_p
                # # # mask_min = torch.quantile(mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2]*mask.shape[3]), 0.01 ,dim=2)
                # mask_min = torch.min(mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2]*mask.shape[3]), dim = 2)[0]
                # # mask_max = torch.quantile(mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2]*mask.shape[3]), 0.995, dim=2)
                # mask_max = torch.max(mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2]*mask.shape[3]), dim = 2)[0]
                # rescaled_mask = (mask - mask_min[:,:,None,None])/(mask_max[:,:,None,None] - mask_min[:,:,None,None])
                # hm_p = rescaled_mask
                # # rescaled_mask = m(mask)
                # # rescaled_mask = mask
                # # rescaled_mask = 
                # # sig_mask = m(8*(rescaled_mask-0.5))
                # # rescaled_mask = m(mask)
                # sig_mask = m(7*(rescaled_mask-0.5))
                # # print('target,',target[0,0,25:38,25:38])
                # # mse_loss = nn.MSELoss()
                # # diff = mse_loss(mask, target).mean()
                # # print('mu_x', mu_x[0])
                # # print('target,',target)
                # # med_filt = medfilt(sig_mask, kernel_size = 5)
                # fg = img[:,:,3:-3,3:-3] * sig_mask
                # bg = img[:,:,3:-3,3:-3] * (1-sig_mask)

                # bg_mean = torch.mean(bg, dim=(2,3))
                # # pure_noise = torch.randn_like(mu_x)*bg_mean[:,:,None,None]
                
                # # print('bg_mean shape', bg_mean.shape, bg_mean[0])
                # fg_mean = torch.mean(fg, dim=(2,3))
                # # print('fg mean shape', fg_mean.shape, fg_mean[0])
                # bg_all = torch.ones(bg.size()).to(self.device)*bg_mean[:,:,None,None]
                # bg_add = bg_all + bg
                # # pure_noise_mask = pure_noise * mask
                # # pure_noise = noise_std * pure_noise
                # # pme_highlight = pme_out * mask
                # # print('mask', mask)
                # # mask = 1-mask
                # fg_noise = fg 
                # # bg_noise = mu_x[:,:,100-32:132,278-32:278+32]
                # bg_noise = bg
                # # pred_out = self.models[Denoiser.PROB_ESTIMATOR](pme_highlight)
                # # mask = pred_out
                # loss_pred = 0
                final_loss = loss_out
                mask_loss = 0
                pred_loss = 0
                consis_loss = 0
                pred_loss_s = 0


            return {
                PipelineOutput.INPUTS: data,
                PipelineOutput.IMG_MU: mu_x,
                PipelineOutput.TARGET: hm,
                PipelineOutput.AUG_LOSS: consis_loss,
                # PipelineOutput.IMG_PME: pme_out,
                PipelineOutput.LOSS: final_loss,
                PipelineOutput.IMG_DENOISED: pme_out,
                PipelineOutput.DETECT_LOSS: pred_loss,
                PipelineOutput.DENOISE_LOSS: loss_out,
                PipelineOutput.NOISE_STD_DEV: noise_std_out,
                PipelineOutput.MODEL_STD_DEV: net_std_out,
                PipelineOutput.DETECT: hm_p,
                PipelineOutput.GT: gt,
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
        noise_style = self.cfg[ConfigValue.NOISE_STYLE]
        # noise_style = "gauss"
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
            # if train:
            # net_out, hm_p = self.models[Denoiser.MODEL](inp)
        # Call the NN with the current image etc.
        net_out, hm_p = self.models[Denoiser.MODEL](noisy_in)
        hm_p = _sigmoid(hm_p)
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
                net_std_out = net_std_out.unsqueeze(0)
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
            # PipelineOutput.DETECT: hm_p,
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
