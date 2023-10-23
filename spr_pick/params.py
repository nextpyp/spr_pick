from __future__ import annotations

from enum import Enum, auto
from typing import List


class NoiseAlgorithm(Enum):
    SELFSUPERVISED_DENOISING = "ssdn"
    SELFSUPERVISED_DENOISING_MEAN_ONLY = "ssdn_u_only"
    NOISE_TO_NOISE = "n2n"
    NOISE_TO_CLEAN = "n2c"
    NOISE_TO_VOID = "n2v"  # Unsupported


class NoiseValue(Enum):
    UNKNOWN_CONSTANT = "const"
    UNKNOWN_VARIABLE = "var"
    KNOWN = "known"

class Loss(Enum):
    FOCAL = "focal"
    MSE = "mse"


class Pipeline(Enum):
    MSE = "mse"
    SSDN = "ssdn"
    MASK_MSE = "mask_mse"


class Blindspot(Enum):
    ENABLED = "blindspot"
    DISABLED = "normal"


class ConfigValue(Enum):
    INFER_CFG = auto()
    ALGORITHM = auto()
    BLINDSPOT = auto()
    PIPELINE = auto()
    IMAGE_CHANNELS = auto()

    NOISE_STYLE = auto()
    BB = auto()
    LEARNING_RATE = auto()
    LR_RAMPUP_FRACTION = auto()
    LR_RAMPDOWN_FRACTION = auto()
    DETECTLOSS = auto()
    NOISE_VALUE = auto()
    DIAGONAL_COVARIANCE = auto()
    NMS = auto()

    EVAL_INTERVAL = auto()
    PRINT_INTERVAL = auto()
    SNAPSHOT_INTERVAL = auto()
    ITERATIONS = auto()
    DATALOADER_WORKERS = auto()
    TRAIN_DATASET_NAME = auto()
    TRAIN_DATASET_TYPE = auto()
    TRAIN_GT_PATH = auto()
    TRAIN_DATA_PATH = auto()
    TRAIN_LABEL_PATH = auto()
    TRAIN_PATCH_SIZE = auto()
    TRAIN_MINIBATCH_SIZE = auto()
    TEST_GT_PATH = auto()
    TEST_DATASET_NAME = auto()
    TEST_DATASET_TYPE = auto()
    TEST_DATA_PATH = auto()
    TEST_LABEL_PATH = auto()
    TEST_MINIBATCH_SIZE = auto()
    PIN_DATA_MEMORY = auto()
    JOINT_LR = 1e-5
    ALPHA = auto()
    FRACTION = auto()
    NUM_EVAL = auto()
    TAU = auto()
class DatasetType(Enum):
    HDF5 = auto()
    FOLDER = auto()
    TXT = auto()


class StateValue(Enum):
    INITIALISED = auto()
    MODE = auto()

    ITERATION = auto()
    REFERENCE = auto()
    HISTORY = auto()


class HistoryValue(Enum):
    TRAIN = auto()
    EVAL = auto()
    TIMINGS = auto()


class PipelineOutput(Enum):
    INPUTS = auto()
    LOSS = "loss"
    DETECT_LOSS = "det_loss"
    DENOISE_LOSS = "denoise_loss"
    IMG_DENOISED = "out"
    IMG_MU = "out_mu"
    NOISE_STD_DEV = "noise_std"
    MODEL_STD_DEV = "model_std"
    TARGET = "target"
    GT = "ground_truth"
    AUG_LOSS = "aug_loss"
    DETECT = auto()
