from __future__ import annotations

import os
import math
import torch
import torch.optim as optim
import re
import glob
import logging
import numpy as np

import spr_pick
from sklearn.cluster import KMeans

from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Sampler
from torchvision.transforms import RandomCrop
import torchvision.transforms as transforms
from spr_pick.datasets import (
    HDF5Dataset,
    UnlabelledImageFolderDataset,
    FixedLengthSampler,
    StratifiedCoordinateSampler,
    NoisyDataset,
    SamplingOrder,
    DetectionDataset,
    MicrographDataset
)

from spr_pick.params import (
    ConfigValue,
    StateValue,
    PipelineOutput,
    Pipeline,
    DatasetType,
    NoiseValue,
    HistoryValue,
)

from spr_pick.models import NoiseNetwork
from spr_pick.models import JointNetwork
# from ssdn.denoiser import Denoiser
from spr_pick.denoiser_v2 import Denoiser
from spr_pick.utils import TrackedTime, Metric, MetricDict, separator
from spr_pick.utils.data_format import DataFormat
from spr_pick.utils.algorithms import non_maximum_suppression, match_coordinates, find_contamination
from spr_pick.cfg import DEFAULT_RUN_DIR
from typing import Dict, Tuple, Union, Callable
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from spr_pick.utils.crop import MyRandomCrop
import pandas as pd
import spr_pick.utils.files as file_utils
import timeit
from scipy.signal import medfilt

logger = logging.getLogger("joint.train")


class DenoiserTrainer:
    """Class that handles training of a Denoiser model using an iteration based
    approach (one image == one iteration). Relies on Dataloaders for data
    preparation for the relevant Denoiser pipleine. Start via `train()` method
    after configuration initialised.

    Call the external  `resume_run()` if resuming an existing training;
    This will create a new DenoiserTrainer().

    Args:
        cfg (Dict): Dictionary with configuration to train on. This dictionary
            will not be assessed for validity until execution.
        state (Dict, optional): Dictionary with training state. Defaults to an
            empty dictionary.
    runs_dir (str, optional): Root directory to create run directory.
        Defaults to DEFAULT_RUN_DIR.
    run_dir (str, optional): Explicit run directory name, will automatically
        generate using configuration if not provided. Defaults to None.
    """

    def __init__(
        self,
        cfg: Dict,
        mode: str,
        state: Dict = {},
        runs_dir: str = DEFAULT_RUN_DIR,
        run_dir: str = None,
        alpha: float = 0.5,
        tau: float = 0.01,
        bb: int = 32,
    ):
        self.runs_dir = os.path.abspath(runs_dir)
        self._run_dir = run_dir
        self._writer = None
        #self.detect = detect
        self.cfg = cfg
        if self.cfg:
            spr_pick.cfg.infer(self.cfg)
        self.state = state
        self.mode = mode
        #self.detect_loss = detect_loss
        #if self.detect_loss is not None:
        self.alpha = alpha
        self.tau = tau
        self.bb = bb
        # self.num_particles = num_particles
        self._denoiser: Denoiser = None
        self._train_iter = None
        self.reduce = False

        self.trainloader, self.trainset, self.train_sampler = None, None, None
        self.testloader, self.testset, self.test_sampler = None, None, None

    @property
    def denoiser(self) -> Denoiser:
        #print('hello dn')
        return self._denoiser

    @denoiser.setter
    def denoiser(self, denoiser: Denoiser):
        #print('hi dn')
        self._denoiser = denoiser
        # Refresh optimiser for parameters
        self.init_optimiser()
    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def init_optimiser(self):
        """Create a new Adam optimiser for the current denoiser parameters.
        Uses defaults except a reduced beta2. When optimiser is accessed via
        its property the learning rate will be updated appropriately for the
        current training state.
        """
        if self.mode == 'joint' or self.mode == 'denoise':
            for name, param in self.denoiser.named_parameters():
                param.requires_grad = True


       
        self._optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.denoiser.parameters()), betas=[0.9, 0.99])

    def new_target(self):
        """Create a new Denoiser to train. Resets training state.
        """
        self.denoiser = Denoiser(self.cfg, mode = self.mode)
        self.init_state()

    def init_state(self):
        """Initialise the state for a fresh training.
        """
        self.state[StateValue.INITIALISED] = True
        self.state[StateValue.ITERATION] = 0
        # History stores events that may happen each iteration
        self.state[StateValue.HISTORY] = {}
        history_state = self.state[StateValue.HISTORY]
        history_state[HistoryValue.TRAIN] = MetricDict()
        history_state[HistoryValue.EVAL] = MetricDict()
        history_state[HistoryValue.TIMINGS] = defaultdict(TrackedTime)
        self.reset_metrics()

    def train(self):
        if self.denoiser is None:
            #print('None')
            self.new_target()
        # print('hello denoiser')
        denoiser = self.denoiser
        # self.denoiser.fill()
        # print('after fill')
        # print(self.denoiser)
        # print(denoiser)
        # Ensure writer is initialised
        _ = self.writer
        spr_pick.logging_helper.setup(self.run_dir_path, "log.txt")
        logger.info(spr_pick.utils.separator())

        logger.info("Loading Training Dataset...")
        self.trainloader, self.trainset, self.train_sampler = self.train_data()
        logger.info("Loaded Training Dataset.")
        if self.cfg[ConfigValue.TEST_DATA_PATH]:
            logger.info("Loading Validation Dataset...")
            self.testloader, self.testset, self.test_sampler = self.test_data()
            logger.info("Loaded Validation Dataset.")

        logger.info(spr_pick.utils.separator())
        logger.info("TRAINING STARTED")
        logger.info(spr_pick.utils.separator())

        # Use history for metric tracking
        history = self.state[StateValue.HISTORY]
        train_history = self.state[StateValue.HISTORY][HistoryValue.TRAIN]

        # Run for trainloader, use internal loop break so that interval checks
        # can run at start of loop
        # 

        if self.mode == "denoise":
            
            iteration = self.state[StateValue.ITERATION]

           
            data_itr = iter(self.trainloader)
            # print(self.train_sampler)
            while True:
                
                iteration = self.state[StateValue.ITERATION]

                if (
                    iteration % self.cfg[ConfigValue.EVAL_INTERVAL] == 0
                ) and self.testloader is not None:
                    torch.cuda.empty_cache()
                    self._evaluate(
                        self.testloader, output_callback=self.validation_output_callback(0),
                    )
                if iteration % self.cfg[ConfigValue.PRINT_INTERVAL] == 0:
                    history[HistoryValue.TIMINGS]["total"].update()
                    last_print = history[HistoryValue.TIMINGS]["last_print"]
                    last_print.update()
                    # Update ETA with metrics captured between prints
                    samples = (
                        history[HistoryValue.EVAL]["n"] + history[HistoryValue.TRAIN]["n"]
                    )
                    self.update_eta(samples, last_print.total)
                    logger.info(self.state_str(eval_prefix="VALID"))
                    self.write_metrics(eval_prefix="valid")
                        # Reset
                    last_print.total = 0
                    self.reset_metrics()
                if iteration % self.cfg[ConfigValue.SNAPSHOT_INTERVAL] == 0:
                    self.snapshot()

                # --- INTERNAL LOOP BREAK --- #
                # if iteration >= 600000:
                #     break
                if iteration >= self.cfg[ConfigValue.ITERATIONS]:
                    break
                # self.alpha_use = self.alpha
                # alpha = 0.0
                # print('trainnnninining')
                data = next(data_itr)
                image_count = data[DetectionDataset.INPUT].shape[0]
                target = data[DetectionDataset.TARGET]
                # Run pipeline calculating gradient from loss
                self.denoiser.unfill()
                self.denoiser.train()
                
                optimizer = self.optimizer
                optimizer.zero_grad()
                # print('train pipeline')
                # pi_use = self.pi - self.positive_fraction
                outputs = denoiser.run_pipeline(data, train = True)
                torch.mean(outputs[PipelineOutput.LOSS]).backward()
                optimizer.step()

                # Increment metrics to be recorded at end of print interval
                with torch.no_grad():
                    train_history["n"] += image_count
                    # print('loss')
                    # print(outputs[PipelineOutput.LOSS])
                    train_history["loss"] += outputs[PipelineOutput.LOSS]
                    # train_history["denoise_loss"] += outputs[PipelineOutput.DENOISE_LOSS]

                    if PipelineOutput.NOISE_STD_DEV in outputs:
                        output = outputs[PipelineOutput.NOISE_STD_DEV]
                        train_history[PipelineOutput.NOISE_STD_DEV.value] += output * 255
                    if PipelineOutput.MODEL_STD_DEV in outputs:
                        output = outputs[PipelineOutput.MODEL_STD_DEV]
                        train_history[PipelineOutput.MODEL_STD_DEV.value] += output * 255

                self.state[StateValue.ITERATION] += image_count
            logger.info(separator())
            logger.info("TRAINING FINISHED")
            logger.info(separator())

            # Save final output weights
            self.snapshot()
            self.snapshot(
                output_name="final-{}.wt".format(self.denoiser.config_name()),
                subdir="",
                model_only=True,
            )

        if self.mode == "joint":
            
            iteration = self.state[StateValue.ITERATION]

           
            data_itr = iter(self.trainloader)
            # print(self.train_sampler)
            while True:
                
                iteration = self.state[StateValue.ITERATION]

                if (
                    iteration % self.cfg[ConfigValue.EVAL_INTERVAL] == 0
                ) and self.testloader is not None:
                    torch.cuda.empty_cache()
                    self._evaluate(
                        self.testloader, output_callback=self.validation_output_callback(0),
                    )
                if iteration % self.cfg[ConfigValue.PRINT_INTERVAL] == 0:
                    history[HistoryValue.TIMINGS]["total"].update()
                    last_print = history[HistoryValue.TIMINGS]["last_print"]
                    last_print.update()
                    # Update ETA with metrics captured between prints
                    samples = (
                        history[HistoryValue.EVAL]["n"] + history[HistoryValue.TRAIN]["n"]
                    )
                    self.update_eta(samples, last_print.total)
                    logger.info(self.state_str(eval_prefix="VALID"))
                    self.write_metrics(eval_prefix="valid")
                        # Reset
                    last_print.total = 0
                    self.reset_metrics()
                if iteration % self.cfg[ConfigValue.SNAPSHOT_INTERVAL] == 0:
                    self.snapshot()

                # --- INTERNAL LOOP BREAK --- #
                # if iteration >= 600000:
                #     break
                if iteration >= self.cfg[ConfigValue.ITERATIONS]:
                    break


                data = next(data_itr)
                image_count = data[DetectionDataset.INPUT].shape[0]
                target = data[DetectionDataset.TARGET]

                # Run pipeline calculating gradient from loss
                self.denoiser.train()
                self.denoiser.unfill()
                optimizer = self.optimizer
                optimizer.zero_grad()
                # pi_use = self.pi - self.positive_fraction
                # self.update_alpha()
                # print('alpha', self.alpha)
                outputs = denoiser.run_pipeline(data,self.alpha, self.tau, train = True)
                torch.mean(outputs[PipelineOutput.LOSS]).backward()
                optimizer.step()

                # Increment metrics to be recorded at end of print interval
                with torch.no_grad():
                    train_history["n"] += image_count
                    train_history["loss"] += outputs[PipelineOutput.LOSS]
                    train_history["denoise_loss"] += outputs[PipelineOutput.DENOISE_LOSS]
                    train_history["detect_loss"] += outputs[PipelineOutput.DETECT_LOSS].unsqueeze(0)
                    train_history["aug_loss"] += outputs[PipelineOutput.AUG_LOSS].unsqueeze(0)

                    # Calculate true PSNR losses for outputs using clean references
                    # Known to be patches so no need to unpad
                    # for key, name in self.img_outputs(prefix="psnr").items():
                    #     train_history[name] += self.calculate_psnr(outputs, key, False)
                    # Track extra metrics if available
                    #print('metrics')
                    if PipelineOutput.NOISE_STD_DEV in outputs:
                        output = outputs[PipelineOutput.NOISE_STD_DEV]
                        train_history[PipelineOutput.NOISE_STD_DEV.value] += output * 255
                    if PipelineOutput.MODEL_STD_DEV in outputs:
                        output = outputs[PipelineOutput.MODEL_STD_DEV]
                        train_history[PipelineOutput.MODEL_STD_DEV.value] += output * 255

                # Progress
                self.state[StateValue.ITERATION] += image_count
            logger.info(separator())
            logger.info("TRAINING FINISHED")
            logger.info(separator())

            # Save final output weights
            self.snapshot()
            self.snapshot(
                output_name="final-{}.wt".format(self.denoiser.config_name()),
                subdir="",
                model_only=True,
            )

    def evaluate(
        self,
        dataloader: DataLoader,
        output_callback: Callable[int, Tensor, None] = None,
    ):
        self.reset_metrics(train=False)
        return self._evaluate(dataloader, output_callback)

    def _evaluate(
        self, dataloader: DataLoader, output_callback: Callable[int, Dict, None],
    ):
        self.denoiser.eval()
        # print('before fill in eval', self.denoiser)
        self.denoiser.fill()
        # print('after fill in eval', self.denoiser)
        with torch.no_grad():
            eval_history = self.state[StateValue.HISTORY][HistoryValue.EVAL]
            idx = 0
            for data in dataloader:

                image_count = data[DetectionDataset.INPUT].shape[0]
                # print('image_count')
                # print(image_count)
                # print('eval pipeline')
                outputs = self.denoiser.run_pipeline(data, train=False)
                eval_history["n"] += image_count
                # Calculate true PSNR losses for outputs using clean references
                metadata = outputs[PipelineOutput.INPUTS][DetectionDataset.METADATA]
                clean = metadata[DetectionDataset.Metadata.GT]
                # print('clean', len(clean))
                if len(clean) > 0:
                    for key, name in self.img_outputs(prefix="psnr").items():
                        # print('key')
                        # print(key)
                        # print(name)
                        # self.calculate_psnr(outputs, key, unpad=True)
                        eval_history[name] += self.calculate_psnr(outputs, key, unpad=True)
                if output_callback:
                    output_callback(idx, outputs)
                idx += image_count
            # print('final idx', idx)

    @property
    def optimizer(self) -> Optimizer:
        """Fetch optimizer whilst applying cosine ramped learning rate. Aim to only
        call this once per iteration to avoid extra computation.

        Returns:
            Optimizer: Adam optimizer with learning rate set automatically.
        """
        learning_rate = self.learning_rate
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = learning_rate
        return self._optimizer

    @property
    def learning_rate(self) -> float:
        
        if self.mode == "joint" or self.mode == "denoise":
            learning_rate = spr_pick.utils.compute_ramped_lrate(
                self.state[StateValue.ITERATION],
                # 20000,
                self.cfg[ConfigValue.ITERATIONS],
                self.cfg[ConfigValue.LR_RAMPDOWN_FRACTION],
                self.cfg[ConfigValue.LR_RAMPUP_FRACTION],
                1e-4,
            )
            return learning_rate

    def validation_output_callback(
        self, output_index: int
    ) -> Callable[int, Dict, None]:
        """Callback that saves only one specific dataset image during each evaluation.

        Args:
            output_index (int): Image index to save.

        Returns:
            Callable[[int, Dict], None]: Callback function for evaluator.
        """

        def callback(output_0_index: int, outputs: Dict):
            inp = outputs[PipelineOutput.INPUTS][DetectionDataset.INPUT]
            output_count = inp.shape[0]
            # If the required image is in this batch export it
            bi = output_index - output_0_index
            if bi >= 0 and bi < output_count:
                output_dir = os.path.join(self.run_dir_path, "val_imgs_"+self.mode)
                fileformat = "{name}_{iter:08}_{desc}.png"
                scoreformat = "{name}_{iter:08}_{desc}.txt"
                if self.mode == "joint":
                    self._save_image_outputs(outputs, output_dir, fileformat, bi, scoreformat)
                else:
                    self._save_image_outputs(outputs, output_dir, fileformat, bi)

        return callback

    def save_image_outputs(
        self,
        outputs: Dict,
        output_dir: str,
        fileformat: str,
        scoreformat: str,
        batch_indexes: int = None,
    ):
        """Save all images present in the outputs. If the data is batched all
        values will saved separately.

        Args:
run_            outputs (Dict): Outputs to extract images from.
            output_dir (str): Output directory for images.
            fileformat (str): Template format for filenames. This can use keyword
                string arguments: (`iter`, TRAIN_ITERATION), (`index`, IMG INDEX),
                (`desc`, IMG DESCRIPTION).
            batch_indexes (int, optional): Batch indexes to save. Defaults to
                None; indicating all.
        """
        if batch_indexes is None:
            metadata = outputs[PipelineOutput.INPUTS][DetectionDataset.METADATA]
            #clean = metadata[NoisyDataset.Metadata.CLEAN]
            # print('hello batch')
            batch_indexes = range(clean.shape[0])
        for bi in batch_indexes:
            self._save_image_outputs(outputs, output_dir, fileformat, bi, scoreformat)

    def _save_image_outputs(
        self, outputs: Dict, output_dir: str, fileformat: str, batch_index: int, scoreformat = None
    ):
        """Save all images present in the outputs for a single item in the batch.

        Args:
            outputs (Dict): Outputs to extract images from.
            output_dir (str): Output directory for images.
            fileformat (str): Template format for filenames. This can use keyword
                string arguments: (`iter`, TRAIN_ITERATION), (`index`, IMG INDEX),
                (`desc`, IMG DESCRIPTION).
            batch_index (int, optional): Item index in batch to save.
        """
        os.makedirs(output_dir, exist_ok=True)
        metadata = outputs[PipelineOutput.INPUTS][DetectionDataset.METADATA]
        # print('metadata name',metadata[DetectionDataset.Metadata.NAME])
        def make_path(desc: str) -> str:
            filename_args = {
                "iter": self.state[StateValue.ITERATION],
                "index": metadata[DetectionDataset.Metadata.INDEXES][batch_index],
                "desc": desc,
                "name": metadata[DetectionDataset.Metadata.NAME][batch_index],
            }

            filename = fileformat.format(**filename_args)
            # print('filename', filename)
            # if scoreformat is not None:
            #     scorename = scoreformat.format(**filename_args)

            return os.path.join(output_dir, filename)
        def score_path(desc:str) -> str:
            filename_args = {
                "iter": self.state[StateValue.ITERATION],
                "index": metadata[DetectionDataset.Metadata.INDEXES][batch_index],
                "desc": desc,
                "name": metadata[DetectionDataset.Metadata.NAME][batch_index],
            }

            filename = scoreformat.format(**filename_args)

            return os.path.join(output_dir, filename)


        def unpad_save(img: Tensor, target: Tensor, desc: str, find_contam = False):
            out = DetectionDataset.unpad(img, target, metadata, batch_index)

            # print('out', out[0].shape)
            # print('target', target.shape)
            spr_pick.utils.save_tensor_image(out[0], make_path(desc))




        def unpad_save_filt(img: Tensor, target: Tensor, desc: str):
            out = DetectionDataset.unpad(img, target, metadata, batch_index)
            spr_pick.utils.save_tensor_image_filt(out[0], make_path(desc))

        def unpad_save_detection(detect: Tensor, target: Tensor, desc: str, contam):
            out = DetectionDataset.unpad(detect, target, metadata, batch_index)

            out_score= out[0][0]
            out_score = out_score.cpu().numpy()
            # print('out_score', out_score.shape)
            x_max, y_max = out_score.shape[0]-30, out_score.shape[1]-30
            score, coords = non_maximum_suppression(out_score, self.cfg[ConfigValue.NMS], contam, 0.02)
            name = metadata[DetectionDataset.Metadata.NAME][batch_index]
            #table = pd.DataFrame({'image_name': name, 'x_coord': coords[:,0], 'y_coord': coords[:,1], 'score': score})
            out_file = open(score_path(desc), 'w')
            print('image_name\tx_coord\ty_coord\tscore', file=out_file)
            for i in range(len(score)):
                if coords[i,1] > 30 and coords[i,1] < x_max and coords[i,0] > 30 and coords[i,0] < y_max:
                    print(name + '\t' + str(coords[i,1]) + '\t' + str(coords[i,0]) + '\t' + str(score[i]), file=out_file)
        
            

        # Save all present outputs
        # if NoisyDataset.Metadata.CLEAN in metadata:
        #     unpad_save(metadata[NoisyDataset.Metadata.CLEAN], "cln")
        if PipelineOutput.INPUTS in outputs:
            unpad_save(outputs[PipelineOutput.INPUTS][DetectionDataset.INPUT], outputs[PipelineOutput.INPUTS][DetectionDataset.TARGET],"nsy")
        if PipelineOutput.IMG_DENOISED in outputs:
            unpad_save(outputs[PipelineOutput.IMG_DENOISED], outputs[PipelineOutput.INPUTS][DetectionDataset.TARGET],"out")
            # print('contam,', contam)
            contam = set()
        if PipelineOutput.IMG_MU in outputs:
            # print('mu', outputs[PipelineOutput.IMG_MU].shape)
            unpad_save(outputs[PipelineOutput.IMG_MU],outputs[PipelineOutput.INPUTS][DetectionDataset.TARGET],"out-mu")
        if PipelineOutput.TARGET in outputs:
            unpad_save(outputs[PipelineOutput.TARGET],outputs[PipelineOutput.INPUTS][DetectionDataset.TARGET],"out-target")
        if PipelineOutput.MODEL_STD_DEV in outputs:
            # N.B Scales and adds channel dimension
            # img = outputs[PipelineOutput.MODEL_STD_DEV][:, None, ...]
            img = outputs[PipelineOutput.MODEL_STD_DEV]
            # print('img', img.shape)
            # img /= 10.0 / 255
            unpad_save(img,outputs[PipelineOutput.INPUTS][DetectionDataset.TARGET], "out-std")
        if PipelineOutput.DETECT in outputs:
            unpad_save(outputs[PipelineOutput.DETECT], outputs[PipelineOutput.INPUTS][DetectionDataset.TARGET], "pred_tar")
            # print('unpad_save', outputs)
            unpad_save_detection(outputs[PipelineOutput.DETECT],outputs[PipelineOutput.INPUTS][DetectionDataset.TARGET],"scores", contam)
            # num_of_detect = outputs[PipelineOutput.DETECT].shape[0]


    def snapshot(
        self, output_name: str = None, subdir: str = None, model_only: bool = False
    ):
        """Save the current Denoiser object with or without all training info.

        Args:
            output_name (str, optional): Fixed name for the output file, will default
                to model_{itertion}.{ext}. {ext} = wt for model weights, and training
                for full training configuration.
            subdir (str, optional): Subdirectory of run directory to store models in.
                Will default to 'models' if only saving the model, 'training'
                otherwise.
            model_only (bool, optional): Whether to only save the model state
                dictionary. Defaults to False.
        """

        if subdir is None and self.mode == "joint":
            subdir = "model_jt" if model_only else "training_jt"
        if subdir is None and self.mode == "denoise":
            subdir = "model_dn" if model_only else "training_dn"
        output_dir = os.path.join(self.run_dir_path, subdir)
        os.makedirs(output_dir, exist_ok=True)
        if model_only:
            if output_name is None:
                iteration = self.state[StateValue.ITERATION]
                output_name = "model_{:08d}.wt".format(iteration)
            torch.save(
                self.denoiser.state_dict(), os.path.join(output_dir, output_name)
            )
        else:
            if output_name is None:
                iteration = self.state[StateValue.ITERATION]
                output_name = "model_{:08d}.training".format(iteration)
            torch.save(self.state_dict(), os.path.join(output_dir, output_name))

    def write_metrics(self, eval_prefix: str = "eval"):
        """Writes the accumulated (mean) of each metric in the evaluation and train
        history dictionaries to the current tensorboard writer. Also tracks the
        learning rate.

        Args:
            eval_prefix (str, optional): Subsection for evaluation metrics to be
                written to in the case there are validation and test sets.
                Defaults to "eval".
        """

        def write_metric_dict(metric_dict: Dict, prefix: str):
            for key, metric in metric_dict.items():
                if isinstance(metric, Metric) and not metric.empty():
                    self.writer.add_scalar(
                        prefix + "/" + key,
                        metric.accumulated(),
                        self.state[StateValue.ITERATION],
                    )

        # Training metrics
        metric_dict = self.state[StateValue.HISTORY][HistoryValue.TRAIN]
        write_metric_dict(metric_dict, "train")
        self.writer.add_scalar(
            "train/learning_rate", self.learning_rate, self.state[StateValue.ITERATION],
        )
        # Eval metrics
        metric_dict = self.state[StateValue.HISTORY][HistoryValue.EVAL]
        write_metric_dict(metric_dict, eval_prefix)

    def state_str(self, eval_prefix: str = "EVAL") -> str:
        """String indicating current training state, displaying all metrics. This displays
        training information and evaluation information if present.

        Args:
            eval_prefix (str, optional): String to put before evaluation lines.
                Defaults to "EVAL" with alignment padding.

        Returns:
            str: String containing all available metrics.
        """
        state_str = self.train_state_str()
        if self.state[StateValue.HISTORY][HistoryValue.EVAL]["n"] > 0:
            prefix = "{:10} {:>5}".format("", eval_prefix)
            state_str = os.linesep.join(
                [state_str, self.eval_state_str(prefix=prefix)]
            )
        return state_str

    def train_state_str(self) -> str:
        """String describing current training state. Gives averages of all accumulated
        metrics and remaining time estimates using tracked times.

        Returns:
            str: Generated description.
        """
        def eta_str():
            timings = self.state[StateValue.HISTORY][HistoryValue.TIMINGS]
            if isinstance(timings.get("eta", None), int):
                eta = timings["eta"]
                if eta < 1:
                    return "<1s"
                else:
                    return spr_pick.utils.seconds_to_dhms(eta)
            else:
                return "???"

        history = self.state[StateValue.HISTORY]
        prefix = "TRAIN"
        summary = "[{:08d}] {:>5} | ".format(self.state[StateValue.ITERATION], prefix)
        metric_str_list = []
        train_metrics = history[HistoryValue.TRAIN]
        for key, metric in train_metrics.items():
            if isinstance(metric, Metric) and not metric.empty():
                metric_str_list += ["{}={:8.2f}".format(key, metric.accumulated())]
        summary += ", ".join(metric_str_list)
        # Add time with ETA
        total_train = history[HistoryValue.TIMINGS]["total"]
        if len(metric_str_list) > 0:
            summary += " | "
        summary += "[{} ~ ETA: {}]".format(
            spr_pick.utils.seconds_to_dhms(total_train.total, trim=False), eta_str(),
        )
        return summary

    def eval_state_str(self, prefix: str = "EVAL") -> str:
        """String giving averages of all accumulated evaluation metrics metrics.

        Args:
            eval_prefix (str, optional): String to put at start of state string.
                Defaults to "EVAL".

        Returns:
            str: Generated string.
        """
        summary = "{} | ".format(prefix)
        metric_str_list = []
        eval_metrics = self.state[StateValue.HISTORY][HistoryValue.EVAL]
        for key, metric in eval_metrics.items():
            if isinstance(metric, Metric) and not metric.empty():
                metric_str_list += ["{}={:8.2f}".format(key, metric.accumulated())]
        summary += ", ".join(metric_str_list)

        return summary

    def reset_metrics(self, eval: bool = True, train: bool = True):
        """Clear any metric value and reset the data count for both the evaluation
        and training metric histories.

        Args:
            eval (bool, optional): Reset the eval metrics. Defaults to True.
            train (bool, optional): Reset the train metrics. Defaults to True.
        """

        def reset_metric_dict(metric_dict: Dict):
            metric_dict["n"] = 0
            for key, metric in metric_dict.items():
                if isinstance(metric, Metric):
                    metric.reset()

        if train:
            reset_metric_dict(self.state[StateValue.HISTORY][HistoryValue.TRAIN])
        if eval:
            reset_metric_dict(self.state[StateValue.HISTORY][HistoryValue.EVAL])

    def img_outputs(self, prefix: str = None) -> Dict:
        """For the current configuration, determine the image outputs that will
        be generated and associate them with a descriptive name for saving/processing.

        Args:
            prefix (str, optional): Prefix to prepend to names. Defaults to None.

        Returns:
            Dict: Dictionary of pipeline output keys to names.
        """
        outputs = {PipelineOutput.IMG_DENOISED: "out"}
        if self.cfg[ConfigValue.PIPELINE] == Pipeline.SSDN:
            outputs.update({PipelineOutput.IMG_MU: "mu_out"})
        if prefix:
            for output, key in outputs.items():
                outputs[output] = "_".join((prefix, key))
        return outputs

    @staticmethod
    def calculate_psnr(
        outputs: Dict, output: PipelineOutput, unpad: bool = True
    ) -> Tensor:
        """Calculate all PSNRs for a given pipleine output, handling batching
        and unpadding.

        Args:
            outputs (Dict): Output dictionary from a pipeline execution.
            output (PipelineOutput): Key to image to process.
            unpad (bool, optional): Whether to unpad. Defaults to True.

        Returns:
            Tensor: PSNR results (one per batch image).
        """
        # Get clean reference
        # print('outputs', outputs)

        metadata = outputs[PipelineOutput.INPUTS][DetectionDataset.METADATA]
        clean = metadata[DetectionDataset.Metadata.GT]
        # print('clean', clean.shape)
        # clean = outputs[PipelineOutput.GT]
        if unpad:
        #     clean = DetectionDataset.unpad(clean, metadata)
            clean = [c.to(outputs[output].device) for c in clean]
            cleaned = DetectionDataset.unpad_img(outputs[output], metadata)
            zipped = zip(cleaned, clean)
            psnrs = map(
                lambda x: spr_pick.utils.calculate_psnr(*x, data_format=DataFormat.CHW),
                zipped,
            )
            psnr = torch.stack(list(psnrs))
            # print(psnr)
        else:
            clean = clean.to(outputs[output].device)
            psnr = spr_pick.utils.calculate_psnr(outputs[output], clean)
        return psnr

    @property
    def writer(self) -> SummaryWriter:
        """The Tensorboard Summary Writer. When this method is first called a new
        SummaryWriter will be created targetting the run directory. Any data present
        in the Tensorboard for the current run (i.e. if resuming) will be removed
        from the current iteration onwards.

        Returns:
            SummaryWriter: Initialised Tensorboard SummaryWriter.
        """
        os.makedirs(self.run_dir_path, exist_ok=True)
        if self._writer is None:
            start_iteration = self.state[StateValue.ITERATION]
            self._writer = SummaryWriter(
                log_dir=self.run_dir_path, purge_step=start_iteration+1
            )
        return self._writer

    @property
    def run_dir_path(self) -> str:
        """
        Returns:
            str: Full path to run directory (`run_dir`) inside runs directory.
        """
        return os.path.join(self.runs_dir, self.run_dir)

    @property
    def run_dir(self) -> str:
        """The run path to use for this run. When this method is first called
        a new directory name will be generated using the next run ID and current
        configuration.

        Returns:
            str: Run directory name, note this is not a full path.
        """
        if self._run_dir is None:
            config_name = self.config_name()
            next_run_id = self.next_run_id()
            run_dir_name = "{:05d}-train-{}".format(next_run_id, config_name)
            self._run_dir = run_dir_name
        return self._run_dir

    def next_run_id(self) -> int:
        """For the run directory, search for any previous runs and return an
        ID that is one greater.

        Returns:
            int: Run ID Number
        """
        # Find existing runs
        run_ids = []
        if os.path.exists(self.runs_dir):
            for run_dir_path, _, _ in os.walk(self.runs_dir):
                run_dir = run_dir_path.split(os.sep)[-1]
                try:
                    run_ids += [int(run_dir.split("-")[0])]
                except Exception:
                    continue
        # Calculate the next run name for the current configuration
        next_run_id = max(run_ids) + 1 if len(run_ids) > 0 else 0
        return next_run_id

    def update_alpha(self):
        # not_update = True

        curr_iter = self.state[StateValue.ITERATION]
        total_iter = self.cfg[ConfigValue.ITERATIONS]
        if curr_iter/total_iter > 0.8 and not self.reduce:
            self.alpha = self.alpha-0.25
            self.reduce = True

    def update_eta(
        self, samples: int, elapsed: float, smoothing_factor: int = 0.95
    ) -> float:
        """Update the tracked ETA based on how many samples have been processed
        over a given time period. Estimated time treats evaluation samples as
        taking the same amount of time as training samples.

        Args:
            samples (int): The number of samples in the processing window.
            elapsed (float): The amount of time elapsed in the processing window.
            smoothing_factor (int, optional): The proportion to use of this value
                against the previous ETA. For frequent calls to `update_eta` use
                a low value, for infrequent use a high value. Defaults to 0.95.

        Returns:
            float: The estimated remaining time in seconds.
        """
        timings = self.state[StateValue.HISTORY][HistoryValue.TIMINGS]
        if samples <= 0:
            return timings["eta"]
        # Time per number of processed samples
        t = elapsed / samples
        # Remaining iterations
        r = self.cfg[ConfigValue.ITERATIONS] - self.state[StateValue.ITERATION]
        # Add on eval iterations
        if self.testloader is not None:
            r += len(self.testloader) * math.ceil(r / self.cfg[ConfigValue.EVAL_INTERVAL])
        new_eta = t * r
        if "eta" not in timings:
            timings["eta"] = new_eta
        else:
            sf = smoothing_factor
            timings["eta"] = sf * new_eta + (1 - sf) * timings["eta"]
        return timings["eta"]

    def config_name(self) -> str:
        """Create a configuration name that identifies the current configuration.

        Returns:
            str: Denoiser config string with training iterations and the datasets
                used appended (trainset-evalset-denoiser_cfg-iterations).
        """
        def iter_str() -> str:
            if self.state[StateValue.ITERATION] > 0:
                # Handle eval only case
                iterations = self.state[StateValue.ITERATION]
            else:
                iterations = self.cfg[ConfigValue.ITERATIONS]
            if iterations >= 1000000:
                return "iter%dm" % (iterations // 1000000)
            elif iterations >= 1000:
                return "iter%dk" % (iterations // 1000)
            else:
                return "iter%d" % iterations

        config_name_lst = [spr_pick.cfg.config_name(self.cfg), iter_str()]
        # print('config_name_lst', config_name_lst)
        if self.cfg.get(ConfigValue.TEST_DATASET_NAME, None) is not None:
            # print('config_name_lst test', config_name_lst)
            config_name_lst = [self.cfg[ConfigValue.TEST_DATASET_NAME]] + config_name_lst
        if self.cfg.get(ConfigValue.TRAIN_DATASET_NAME, None) is not None:
            # print('config_name_lst train', config_name_lst)
            config_name_lst = [self.cfg[ConfigValue.TRAIN_DATASET_NAME]] + config_name_lst

        config_name_lst += [str(self.cfg[ConfigValue.ALPHA])]
        config_name_lst += [str(self.cfg[ConfigValue.TAU])]

        config_name_lst += [self.mode]

        # print('config', config_name_lst)
        config_name = "-".join(config_name_lst)
        return config_name

    def state_dict(self) -> Dict:
        """A copy of the target denoiser state as well as all data required to continue
        training from the current point.

        Returns:
            Dict: Resumeable state dictionary.
        """
        state_dict = {}
        state_dict["denoiser"] = self.denoiser.state_dict()
        state_dict["state"] = self.state
        # print('state_Dict')
        # print(self.train_sampler.last_iter())
        # state_dict["train_order_iter"] = self.train_sampler.last_iter().state_dict()

        # Reset train order iterator to match the amount of data actually processed
        # not just the amount of data loaded
        # state_dict["train_order_iter"]["index"] = self.state[StateValue.ITERATION]
        state_dict["optimizer"] = self.optimizer.state_dict()
        state_dict["rng"] = torch.get_rng_state()
        return state_dict

    def load_state_dict(self, state_dict: Union[Dict, str]):
        """Load the contents of a state dictionary into the current instance such that
        training can be resumed when `train()` is called.

        Args:
            state_dict (str): Either a state dictionary or a path to a state dictionary.
                If a string is provided this will be used to load the state dictionary
                from disk.
        """
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict, map_location="cpu")
        self.denoiser = Denoiser.from_state_dict(state_dict["denoiser"], mode = self.mode)
        self.cfg = self.denoiser.cfg
        self.state = state_dict["state"]
        # self._train_iter = SamplingOrder.from_state_dict(state_dict["train_order_iter"])
        # self._optimizer.load_state_dict(state_dict["optimizer"])
        torch.set_rng_state(state_dict["rng"])

    def load_state_dict_detection(self, state_dict: Union[Dict, str]):
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict, map_location="cpu")
        self.denoiser = Denoiser.from_state_dict(state_dict["denoiser"], mode = self.mode)
        self.cfg = self.denoiser.cfg
        torch.set_rng_state(state_dict["rng"])
        self.init_state()
    def load_state_dict_ssdet(self, state_dict: Union[Dict, str]):
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict, map_location="cpu")
        self.denoiser = Denoiser.from_state_dict(state_dict["denoiser"], mode = self.mode)
        self.cfg = self.denoiser.cfg 
        self._optimizer.load_state_dict(state_dict["optimizer"])
        torch.set_rng_state(state_dict["rng"])
        self.init_state()

    def train_data(self) -> Tuple[DataLoader, DetectionDataset, Sampler]:
        """Configure the training set using the current configuration.

        Returns:
            Tuple[Dataset, DataLoader, Sampler]: Returns a NoisyDataset object
                wrapping either a folder or HDF5 dataset, a DataLoader for that
                dataset that uses a FixedLengthSampler (also returned).
        """
        cfg = self.cfg
        if self.mode == 'denoise' or self.mode == 'joint':
            transform = MyRandomCrop(
                cfg[ConfigValue.TRAIN_PATCH_SIZE],
                pad_if_needed=True, labeled_only=True, ss_mode = False, padding_mode = "reflect")
            train=True
            augment = transforms.RandomHorizontalFlip(p=0.5)

        # Load dataset
        if cfg[ConfigValue.TRAIN_DATASET_TYPE] == DatasetType.FOLDER:
            dataset = UnlabelledImageFolderDataset(
                cfg[ConfigValue.TRAIN_DATA_PATH],
                channels=cfg[ConfigValue.IMAGE_CHANNELS],
                transform=transform,
                recursive=True,
            )
        elif cfg[ConfigValue.TRAIN_DATASET_TYPE] == DatasetType.HDF5:
            # It is assumed that the created dataset does not violate the minimum patch size
            dataset = HDF5Dataset(
                cfg[ConfigValue.TRAIN_DATA_PATH],
                transform=transform,
                channels=cfg[ConfigValue.IMAGE_CHANNELS],
            )
        elif cfg[ConfigValue.TRAIN_DATASET_TYPE] == DatasetType.TXT:
            dataset = MicrographDataset(
                cfg[ConfigValue.TRAIN_DATA_PATH],
                cfg[ConfigValue.TRAIN_LABEL_PATH],
                radius = 3,
                train = train,
                crop = cfg[ConfigValue.TRAIN_PATCH_SIZE],
                transform = transform,
                gt_path = cfg[ConfigValue.TRAIN_GT_PATH],
                augment = augment,
                channels=cfg[ConfigValue.IMAGE_CHANNELS],
                bb = cfg[ConfigValue.BB],
                )

        else:
            raise NotImplementedError("Dataset type not implemented")

        dataset = DetectionDataset(
            dataset,
            pad_uniform=False,
            pad_multiple=JointNetwork.input_wh_mul(),
            square=cfg[ConfigValue.BLINDSPOT])
        # Ensure dataset initialised by loading first bit of data
        # _ = dataset[0]
        # pi = dataset.pi
        # positive_fraction = dataset.positive_fraction
        # Create a dataloader that will sample from this dataset for a fixed number of samples
        # sampler = FixedLengthSampler(
        #     dataset, num_samples=self.cfg[ConfigValue.ITERATIONS], shuffled=True,
        # )
        labels = dataset.train_targets
        sampler = StratifiedCoordinateSampler(labels, size=self.cfg[ConfigValue.ITERATIONS]*self.cfg[ConfigValue.TRAIN_MINIBATCH_SIZE], balance=0.1)
        # Resume train sampler
        # print('self._train_iter')
        # print(self._train_iter)
        # if self._train_iter is not None:
        #     sampler.for_next_iter(self._train_iter)
        #     self._train_iter = None

        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=cfg[ConfigValue.TRAIN_MINIBATCH_SIZE],
            num_workers=cfg[ConfigValue.DATALOADER_WORKERS],
            pin_memory=cfg[ConfigValue.PIN_DATA_MEMORY],
        )
        return dataloader, dataset, sampler

    def set_train_data(self, path: str):
        self.cfg[ConfigValue.TRAIN_DATA_PATH] = path
        self.cfg[ConfigValue.TRAIN_DATASET_TYPE] = None
        # self.cfg[ConfigValue.TRAIN_DATASET_NAME] = None
        spr_pick.cfg.infer_datasets(self.cfg)

    def set_train_label(self, path: str):
        self.cfg[ConfigValue.TRAIN_LABEL_PATH] = path 

    def set_train_gt_data(self, path:str):
        self.cfg[ConfigValue.TRAIN_GT_PATH] = path 
    def set_test_gt_data(self, path:str):
        self.cfg[ConfigValue.TEST_GT_PATH] = path

    

    def set_test_label(self, path: str):
        self.cfg[ConfigValue.TEST_LABEL_PATH] = path




    def test_data(self) -> Tuple[DataLoader, DetectionDataset, Sampler]:
        """Configure the test set using the current configuration.

        Returns:
            Tuple[Dataset, DataLoader, Sampler]: Returns a NoisyDataset object
                wrapping either a folder or HDF5 dataset, a DataLoader for that
                dataset that uses a FixedLengthSampler (also returned).
        """
        cfg = self.cfg
        train=False
        # train=True
        # transform = MyRandomCrop(
        #         cfg[ConfigValue.TRAIN_PATCH_SIZE],
        #         pad_if_needed=True, labeled_only=True, ss_mode = False, padding_mode = "reflect")
        # train=True
        # if self.mode == 'ssdet' or self.mode == 'joint':
        #     transform = MyRandomCrop(
        #         cfg[ConfigValue.TRAIN_PATCH_SIZE],
        #         pad_if_needed=True, labeled_only=True, ss_mode = False, padding_mode = "reflect")
        #     train=True
        #     augment = transforms.RandomHorizontalFlip(p=0.5)
       # Load dataset
        if cfg[ConfigValue.TEST_DATASET_TYPE] == DatasetType.FOLDER:
            dataset = UnlabelledImageFolderDataset(
                cfg[ConfigValue.TEST_DATA_PATH],
                recursive=True,
                channels=cfg[ConfigValue.IMAGE_CHANNELS],
            )
        elif cfg[ConfigValue.TEST_DATASET_TYPE] == DatasetType.HDF5:
            dataset = HDF5Dataset(
                cfg[ConfigValue.TEST_DATA_PATH],
                channels=cfg[ConfigValue.IMAGE_CHANNELS],
            )
        elif cfg[ConfigValue.TRAIN_DATASET_TYPE] == DatasetType.TXT:
            # dataset = MicrographDataset(
            #     cfg[ConfigValue.TRAIN_DATA_PATH],
            #     cfg[ConfigValue.TRAIN_LABEL_PATH],
            #     radius = 15,
            #     train = train,
            #     transform = transform,
            #     gt_path = cfg[ConfigValue.TRAIN_GT_PATH],
            #     channels=cfg[ConfigValue.IMAGE_CHANNELS],
            #     )
            # print('test wtf')
            dataset = MicrographDataset(
                cfg[ConfigValue.TEST_DATA_PATH],
                cfg[ConfigValue.TEST_LABEL_PATH],
                radius = 3,
                train = train,
                gt_path = cfg[ConfigValue.TEST_GT_PATH],
                channels=cfg[ConfigValue.IMAGE_CHANNELS],
                bb = cfg[ConfigValue.BB],
                )
        else:
            raise NotImplementedError("Dataset type not implemented")

        dataset = DetectionDataset(
            dataset,
            pad_uniform=False,
            pad_multiple=NoiseNetwork.input_wh_mul(),
            square=cfg[ConfigValue.BLINDSPOT])
        # Ensure dataset initialised by loading first bit of data
        _ = dataset[0]
        # Create a dataloader that will sample from this dataset for a fixed number of samples
        sampler = FixedLengthSampler(
            dataset,
            num_samples=spr_pick.cfg.test_length(cfg),
            shuffled=False,
        )
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=cfg[ConfigValue.TEST_MINIBATCH_SIZE],
            num_workers=cfg[ConfigValue.DATALOADER_WORKERS],
            pin_memory=cfg[ConfigValue.PIN_DATA_MEMORY],
        )
        return dataloader, dataset, sampler

    def set_test_data(self, path: str):
        self.cfg[ConfigValue.TEST_DATA_PATH] = path
        self.cfg[ConfigValue.TEST_DATASET_TYPE] = None
        # self.cfg[ConfigValue.TEST_DATASET_NAME] = None
        spr_pick.cfg.infer_datasets(self.cfg)


def resume_run(run_dir: str, iteration: int = None) -> DenoiserTrainer:
    """Resume training of a Denoiser model from a previous run.

    Args:
        run_dir (str): The root directory of execution. Resumable training states
            are expected to be in {run_dir}/training.
        iteration (int, optional): The iteration to resume from, if not provided
            the last iteration found will be used.

    Returns:
        DenoiserTrainer: Fully initialised trainer which will update existing
            run directory.
    """
    run_dir = os.path.abspath(run_dir)
    runs_dir = os.path.abspath(os.path.join(run_dir, ".."))
    iterations = {}
    for path in glob.glob(os.path.join(run_dir, "training_jt", "*.training")):
        try:
            iterations[int(re.findall(r"\d+", os.path.basename(path))[0])] = path
        except Exception:
            continue
    if iteration is None:
        if len(iterations) == 0:
            raise ValueError("Run directory contains no training files.")
        iteration = max(iterations.keys())

    load_file_path = iterations[iteration]
    logger.info("Loading from '{}'...".format(load_file_path))
    trainer = DenoiserTrainer(
        None, runs_dir=runs_dir, run_dir=os.path.basename(run_dir)
    )
    trainer.load_state_dict(load_file_path)
    logger.info("Loaded training state.")
    # Cannot trust old absolute times so discard
    for timing in trainer.state[StateValue.HISTORY][HistoryValue.TIMINGS].values():
        if isinstance(timing, TrackedTime):
            timing.forget()

    return trainer
