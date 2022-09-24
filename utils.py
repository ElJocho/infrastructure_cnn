"""
This script holds some helping functions which are called within other scripts
"""
import json
import os
from itertools import product
from typing import Dict

import numpy as np
import rasterio
import requests
from rasterio import windows

from definitions import OHSOME_API, RESULT_PATH, logger
from main import TARGET_SIZE


def query(request: Dict, bpolys: str, properties: str = None) -> Dict:
    """This function executes the ohsome query"""
    """Query ohsome API endpoint with filter"""
    url = OHSOME_API + request["endpoint"]
    if properties is not None:
        data = {"bpolys": bpolys, "filter": request["filter"], "properties": properties}
    else:
        data = {"bpolys": bpolys, "filter": request["filter"]}
    logger.info("Query ohsome API.")
    logger.info("Query URL: " + url)
    logger.info("Query Filter: " + request["filter"])

    response = requests.post(url, data=data)

    if response.status_code == 200:
        logger.info("Query successful!")
    elif response.status_code == 404:
        logger.info("Query failed!")
    else:
        logger.info(response.status_code)
    return response.json()


def update_json(key, val) -> None:
    """This function adds a key-value-pair to the metrics-json and creates it if it does
        not exist"""
    if os.path.exists(os.path.join(RESULT_PATH, "metrics.json")):
        with open(os.path.join(RESULT_PATH, "metrics.json"), "r") as file:
            metrics = json.load(file)
    else:
        metrics = {}
    metrics[key] = val
    with open(os.path.join(RESULT_PATH, "metrics.json"), "w") as file:
        json.dump(metrics, file)


def get_tiles(ds, width=256, height=256) -> tuple:
    """This function gets a tile and it's transformation info from a raster"""
    ncols, nrows = ds.meta["width"], ds.meta["height"]
    offsets = product(range(0, ncols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)
    for col_off, row_off in offsets:
        window = windows.Window(
            col_off=col_off, row_off=row_off, width=width, height=height
        ).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform


def write_raster_window(raster, r_mask, window, transform, path, counter) -> bool:
    """This function writes a tile of a image raster and its belonging mask raster into
        a directory (train, test, validation) and pads them to the target size if
        necessary"""
    tiledata_img = raster.read(window=window)
    tiledata_mask = r_mask.read(window=window)
    if (len(tiledata_img.shape) == 3 and np.sum(tiledata_img) == 0) or np.sum(
        tiledata_mask[0]
    ) == 0:
        return False

    if tiledata_img.shape[1] < TARGET_SIZE[0]:
        t = TARGET_SIZE[0] - tiledata_img.shape[1]
        tiledata_img = np.pad(tiledata_img, ((0, 0), (0, t), (0, 0)), constant_values=0)
        tiledata_mask = np.pad(
            tiledata_mask, ((0, 0), (0, t), (0, 0)), constant_values=0
        )
    if tiledata_img.shape[2] < TARGET_SIZE[1]:
        t = TARGET_SIZE[1] - tiledata_img.shape[2]
        tiledata_img = np.pad(tiledata_img, ((0, 0), (0, 0), (0, t)), constant_values=0)
        tiledata_mask = np.pad(
            tiledata_mask, ((0, 0), (0, 0), (0, t)), constant_values=0
        )

    meta_tile_img = raster.meta.copy()
    meta_tile_img.update(
        {
            "height": TARGET_SIZE[0],
            "width": TARGET_SIZE[1],
            "transform": transform,
        }
    )

    meta_tile_mask = r_mask.meta.copy()
    meta_tile_mask.update(
        {
            "height": TARGET_SIZE[0],
            "width": TARGET_SIZE[1],
            "transform": transform,
        }
    )

    with rasterio.open(
        os.path.join(path, f"img/sample_{counter}.tif"), "w", **meta_tile_img
    ) as dest:
        dest.write(tiledata_img)

    with rasterio.open(
        os.path.join(path, f"mask/sample_{counter}.tif"), "w", **meta_tile_mask
    ) as dest:
        dest.write(tiledata_mask)

    return True
