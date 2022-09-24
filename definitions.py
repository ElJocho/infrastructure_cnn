"""
Global Variables and Functions.
"""

import logging.config
import os
from pathlib import Path

import yaml

OHSOME_API = os.getenv("OHSOME_API", default="https://api.ohsome.org/v1/")
DATA_PATH = "./data"
RASTER_PATH = os.path.join(DATA_PATH, "aoi.tif")

PREDICT_PATH = os.path.join(DATA_PATH, "predict")
TRAINING_PATH = os.path.join(DATA_PATH, "training_data")
TRAINING_PATH_IMG = os.path.join(TRAINING_PATH, "img")
TRAINING_PATH_MASK = os.path.join(TRAINING_PATH, "mask")
TEST_PATH = os.path.join(DATA_PATH, "test_data")
TEST_PATH_IMG = os.path.join(TEST_PATH, "img")
TEST_PATH_MASK = os.path.join(TEST_PATH, "mask")
VALIDATION_PATH = os.path.join(DATA_PATH, "valid_data")
VALIDATION_PATH_IMG = os.path.join(VALIDATION_PATH, "img")
VALIDATION_PATH_MASK = os.path.join(VALIDATION_PATH, "mask")
INTERMEDIATE_PATH = os.path.join(DATA_PATH, "intermediate_result")
RESULT_PATH = os.path.join(DATA_PATH, "result")
paths = [
    DATA_PATH,
    TRAINING_PATH,
    TRAINING_PATH_IMG,
    TRAINING_PATH_MASK,
    TEST_PATH,
    TEST_PATH_IMG,
    TEST_PATH_MASK,
    INTERMEDIATE_PATH,
    RESULT_PATH,
    VALIDATION_PATH,
    VALIDATION_PATH_IMG,
    VALIDATION_PATH_MASK,
]

for path in paths:
    Path(path).mkdir(parents=True, exist_ok=True)


def get_logger():
    logs_path = os.path.join(DATA_PATH, "logs")
    logging_file_path = os.path.join(logs_path, "infra_net.log")
    logging_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logging.yaml"
    )
    Path(logs_path).mkdir(parents=True, exist_ok=True)

    with open(logging_config_path, "r") as f:
        logging_config = yaml.safe_load(f)
    logging_config["handlers"]["file"]["filename"] = logging_file_path
    logging.config.dictConfig(logging_config)

    return logging.getLogger("infra_net")


logger = get_logger()
