import os

from spr_pick.params import ConfigValue, DatasetType, NoiseAlgorithm, Pipeline, Loss
from typing import Dict


DEFAULT_RUN_DIR = "hi_runs"


def base():
    return {
        ConfigValue.ITERATIONS: 200000,
        ConfigValue.DETECTLOSS: None,
        ConfigValue.TRAIN_MINIBATCH_SIZE: 16,
        ConfigValue.TEST_MINIBATCH_SIZE: 1,
        ConfigValue.IMAGE_CHANNELS: 1,
        ConfigValue.TRAIN_PATCH_SIZE: 64,
        ConfigValue.LEARNING_RATE: 1e-5,
        ConfigValue.LR_RAMPDOWN_FRACTION: 0.7,
        ConfigValue.LR_RAMPUP_FRACTION: 0.2,
        ConfigValue.EVAL_INTERVAL: 3200,
        ConfigValue.PRINT_INTERVAL: 1280,
        ConfigValue.SNAPSHOT_INTERVAL: 3200,
        ConfigValue.DATALOADER_WORKERS: 4,
        ConfigValue.PIN_DATA_MEMORY: False,
        ConfigValue.DIAGONAL_COVARIANCE: False,
        ConfigValue.TRAIN_DATA_PATH: None,
        ConfigValue.TRAIN_GT_PATH:None,
        ConfigValue.TRAIN_LABEL_PATH: None,
        ConfigValue.TRAIN_DATASET_TYPE: None,
        ConfigValue.TEST_DATA_PATH: None,
        ConfigValue.TEST_LABEL_PATH: None,
        ConfigValue.TEST_GT_PATH: None,
        ConfigValue.TEST_DATASET_TYPE: None,
        ConfigValue.JOINT_LR: 1e-5,
        ConfigValue.ALPHA: 0.8,
        ConfigValue.NMS: 15,
        ConfigValue.NUM_EVAL:1,
        ConfigValue.NOISE_STYLE: None,
        ConfigValue.TAU:0.01,
        ConfigValue.BB: 24,

    }


class DatasetName:
    BSD = "bsd"
    IMAGE_NET = "ilsvrc"
    KODAK = "kodak"
    SET14 = "set14"
    MRC = "mrc"
    LIP = "lip"
    EM10492 = "10492"
    EM10304 = "10304"
    EM10499 = "10499"
    EM10250 = "10250"
    SIG = "sig"
    EM10931 = "10931"
    EM10249 = "10249"
    ERICA = "Erica"
    SIM = "simulated"
    EM10215 = "10215"


def infer_datasets(cfg: Dict):
    """For training and test dataset parameters infer from the path the name of
    the dataset being targetted and whether or not the data should be loaded as
    a h5 file or a folder.

    Args:
        cfg (Dict): Configuration to infer for.
    """

    def infer_dname(path: str):
        # Look for part of dataset name in path for guessing dataset
        dataset_dict = {
            "BSDS300": DatasetName.BSD,
            "ILSVRC": DatasetName.IMAGE_NET,
            "KODAK": DatasetName.KODAK,
            "SET14": DatasetName.SET14,
            "10492": DatasetName.EM10492,
            "LIP": DatasetName.LIP,
            "10304":DatasetName.EM10304,
            "10499": DatasetName.EM10499,
            "10215": DatasetName.EM10215,   
            "MRC": DatasetName.MRC,
            "10250":DatasetName.EM10250,
            "SIG":DatasetName.SIG,
            "10931":DatasetName.EM10931,
            "10249":DatasetName.EM10249,
            "ERICA":DatasetName.ERICA,
            "SIM": DatasetName.SIM

        }
        potentials = []
        for key, name in dataset_dict.items():
            if key.lower() in path.lower():
                potentials += [name]
        if len(potentials) == 0:
            raise ValueError("Could not infer dataset from path.")
        if len(potentials) > 1:
            raise ValueError("Matched multiple datasets with dataset path.")
        return potentials[0]

    def infer_dtype(path: str):
        # Treat files as HDF5 and directories as folders
        #dtype = DatasetType.FOLDER if os.path.isdir(path) else DatasetType.HDF5
        if os.path.isdir(path):
            dtype = DatasetType.FOLDER
        elif path.endswith('.txt'):
            dtype = DatasetType.TXT 
        else:
            dtype = DatasetType.HDF5
        return dtype

    # Infer for training set
    if cfg.get(ConfigValue.TRAIN_DATA_PATH, None) is not None:
        if cfg.get(ConfigValue.TRAIN_DATASET_TYPE, None) is None:
            cfg[ConfigValue.TRAIN_DATASET_TYPE] = infer_dtype(
                cfg[ConfigValue.TRAIN_DATA_PATH]
            )
    # Infer for testing/validation set
    if cfg.get(ConfigValue.TEST_DATA_PATH, None) is not None:
        if cfg.get(ConfigValue.TEST_DATASET_TYPE, None) is None:
            cfg[ConfigValue.TEST_DATASET_TYPE] = infer_dtype(
                cfg[ConfigValue.TEST_DATA_PATH]
            )


def test_length(cfg: Dict) -> int:
    """To give meaningful PSNR results similar amounts of data should be evaluated.
    Return the test length based on image size and image count. Note that for all
    datasets it is assumed the test dataset is being used.

    Args:
        dataset_name (str): Name of the dataset (BSD...),

    Returns:
        int: Image count to test for. When higher than the dataset length existing
            images should be reused.
    """

    return cfg[ConfigValue.NUM_EVAL]


def infer_pipeline(algorithm: NoiseAlgorithm) -> Pipeline:
    if algorithm in [NoiseAlgorithm.SELFSUPERVISED_DENOISING]:
        return Pipeline.SSDN
    elif algorithm in [
        NoiseAlgorithm.SELFSUPERVISED_DENOISING_MEAN_ONLY,
        NoiseAlgorithm.NOISE_TO_NOISE,
        NoiseAlgorithm.NOISE_TO_CLEAN,
    ]:
        return Pipeline.MSE
    elif algorithm in [NoiseAlgorithm.NOISE_TO_VOID]:
        return Pipeline.MASK_MSE
    else:
        raise NotImplementedError("Algorithm does not have a default pipeline.")


def infer_blindspot(algorithm: NoiseAlgorithm):
    if algorithm in [
        NoiseAlgorithm.SELFSUPERVISED_DENOISING,
        NoiseAlgorithm.SELFSUPERVISED_DENOISING_MEAN_ONLY,
    ]:
        return True
    elif algorithm in [
        NoiseAlgorithm.NOISE_TO_NOISE,
        NoiseAlgorithm.NOISE_TO_CLEAN,
        NoiseAlgorithm.NOISE_TO_VOID,
    ]:
        return False
    else:
        raise NotImplementedError("Not known if algorithm requires blindspot.")


def infer(cfg: Dict, model_only: bool = False) -> Dict:
    if cfg.get(ConfigValue.PIPELINE, None) is None:
        cfg[ConfigValue.PIPELINE] = infer_pipeline(cfg[ConfigValue.ALGORITHM])
    if cfg.get(ConfigValue.BLINDSPOT, None) is None:
        cfg[ConfigValue.BLINDSPOT] = infer_blindspot(cfg[ConfigValue.ALGORITHM])

    if not model_only:
        infer_datasets(cfg)
    return cfg


def config_name(cfg: Dict) -> str:
    cfg = infer(cfg)
    config_lst = [cfg[ConfigValue.ALGORITHM].value]

    # Check if pipeline cannot be inferred
    inferred_pipeline = infer_pipeline(cfg[ConfigValue.ALGORITHM])
    if cfg[ConfigValue.PIPELINE] != inferred_pipeline:
        config_lst += [cfg[ConfigValue.PIPELINE].value + "_pipeline"]

    # Add noise information
    config_lst += [cfg[ConfigValue.NOISE_STYLE]]


    config_name = "-".join(config_lst)
    return config_name
