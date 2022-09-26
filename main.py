"""
This script sets preprocessing- and CNN-training variables and starts a run of either
the whole or parts of the workflow
"""
from definitions import logger

# SCRIPT SETTINGS:
DUAL_SHAPES = True
# the tile size which should be processed at once in preprocessing -> set this value based on your RAM. 16 GB atm.
TILE_WIDTH = 12800
TILE_HEIGHT = 12800


# ML VARIABLES:

EPOCH = 50  # more is better
BATCH_SIZE = 10
TARGET_SIZE = [256, 256]


def main(mode: str) -> None:
    if mode == "Preprocessing":
        import preprocessing
        logger.info("Working Mode: Preprocess the data")
        preprocessing.preprocessing_data()
    elif mode == "unet":
        import unet

        logger.info("Working Mode: Train the model and predict")
        unet.unet_execution()
    elif mode == "Complete":
        import preprocessing
        import unet

        logger.info(
            "Working Mode: Complete run, including preprocessing training and predicting"
        )

        logger.info("Doing the preprocessing")
        preprocessing.preprocessing_data()
        logger.info("Training the model and predict the raster")
        unet.unet_execution()

    elif mode == "Predict":
        import unet
        unet.predict_raster()
    else:
        raise ValueError("Working mode must be Complete, unet or Preprocessing.")

if __name__ == "__main__":
    working_mode = "Complete"
    main(working_mode)
