"""
This script creates training, test, and validation data based on the input raster and
OSM data
"""
import json
import logging
import math
import os

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio import mask
from shapely.geometry import box
from skimage.filters import median
from skimage.morphology import disk
from tqdm import tqdm

from definitions import (
    INTERMEDIATE_PATH,
    RASTER_PATH,
    TEST_PATH,
    TRAINING_PATH,
    VALIDATION_PATH,
    logger,
)
from main import TARGET_SIZE, TILE_HEIGHT, TILE_WIDTH, DUAL_SHAPES
from utils import get_tiles, query, write_raster_window

# defining the ohsomeAPI queries
buil_def = {
    "description": "All buildings in an area",
    "endpoint": "elements/geometry",
    "filter": """
         building = * and geometry:polygon and building != roof
    """,
}

small_road_def = {
    "description": "All 'smaller' roads in an area",
    "endpoint": "elements/geometry",
    "filter": """
         highway in (residential, motorway_link, trunk_link, primary_link, secondary_link, tertiary_link) and geometry:line
    """,
}

big_road_def = {
    "description": "All 'bigger' roads in an area",
    "endpoint": "elements/geometry",
    "filter": """
         highway in (motorway, trunk, primary, secondary, tertiary) and geometry:line
    """,
}


def get_building_data(raster) -> gpd.GeoDataFrame:
    """
    This function queries OSM buildings within the bounding box of the raster and
        reprojects them into the rasters' projection
    :param raster: the input raster which is used for ML
    :return: geopandas' dataframe of the OSM buildings
    """
    bbox = box(*raster.bounds)
    bbox = gpd.GeoSeries([bbox]).set_crs(raster.crs).to_crs(epsg=4326).__geo_interface__
    bbox = json.dumps(bbox)
    buildings = query(buil_def, bbox)
    buildings = (
        gpd.GeoDataFrame.from_features(buildings["features"])
        .set_crs(epsg=4326)
        .to_crs(crs=raster.crs)
    )
    buildings.to_file(
        os.path.join(INTERMEDIATE_PATH, "buildings.geojson"), driver="GeoJSON"
    )
    return buildings


def get_highway_data(raster) -> gpd.GeoDataFrame:
    """
    This function queries OSM highways within the bounding box of the raster and
        reprojects them into the rasters' projection
    :param raster: the input raster which is used for ML
    :return: geopandas' dataframe of the OSM buildings
    """
    bbox = box(*raster.bounds)
    bbox = gpd.GeoSeries([bbox]).set_crs(raster.crs).to_crs(epsg=4326).__geo_interface__
    bbox = json.dumps(bbox)
    small_roads = query(small_road_def, bbox)
    big_roads = query(big_road_def, bbox)

    small_roads = (
        gpd.GeoDataFrame.from_features(small_roads["features"])
        .set_crs(epsg=4326)
        .to_crs(crs=raster.crs)
    )
    big_roads = (
        gpd.GeoDataFrame.from_features(big_roads["features"])
        .set_crs(epsg=4326)
        .to_crs(crs=raster.crs)
    )
    small_roads.geometry = small_roads.buffer(3.66)
    big_roads.geometry = big_roads.buffer(5)
    roads = gpd.GeoDataFrame(pd.concat([big_roads, small_roads], ignore_index=True))
    roads.to_file(
        os.path.join(INTERMEDIATE_PATH, "roads.geojson"), driver="GeoJSON"
    )
    return roads


def generate_mask(raster, vector, second_vector=None) -> None:
    """
    This function generates a binary mask for a raster indicating where vector features
        overlay the raster
    :param raster: the raster the mask shoul be generated for
    :param vector: the vector features which should be used for mask creating
    :return: None. The mask will be saved as file to the harddrive in order to use
        tile-by-tile creation
    """
    if os.path.exists(os.path.join(INTERMEDIATE_PATH, "masked_raster.tif")):
        os.remove(os.path.join(INTERMEDIATE_PATH, "masked_raster.tif"))
    out_meta = raster.meta.copy()

    out_meta.update(
        {
            "driver": "GTiff",
            "count": int(1 if second_vector is None else 3),
        }
    )

    tiles_needed = math.ceil(
        (out_meta["width"] * out_meta["height"]) / (TILE_WIDTH * TILE_HEIGHT)
    )
    # change logging level as rasterio always warns when using BIGTIFF option in "r+"
    rasterio_logger = rasterio.logging.getLogger()
    rasterio_logger.setLevel(logging.ERROR)
    logger.setLevel(logging.ERROR)
    for window, transform in tqdm(
        # get a tile of the original raster
        get_tiles(raster, TILE_WIDTH, TILE_HEIGHT),
        total=tiles_needed,
    ):
        meta_tile = raster.meta.copy()
        meta_tile["transform"] = transform
        meta_tile["width"], meta_tile["height"] = window.width, window.height

        tiledata = raster.read(window=window)
        with rasterio.open(
            os.path.join(INTERMEDIATE_PATH, f"tile.tif"), "w", **meta_tile
        ) as dest:
            dest.write(tiledata)
        with rasterio.open(os.path.join(INTERMEDIATE_PATH, f"tile.tif"), "r") as tile:
            out_image, out_transform = mask.mask(
                tile,
                vector,
                nodata=0,
                all_touched=False,
                invert=False,
                filled=True,
                crop=False,
                pad=False,
                pad_width=0.5,
                indexes=None,
            )
        # create single band from triple band and classify binary
        out_image = out_image[0]
        out_image[out_image > 0] = 1

        # reduce salt-n-pepper noise
        out_image = median(out_image, disk(1), mode="constant", cval=0)

        if second_vector is None:
            out_image = np.array([out_image])

        if second_vector is not None:
            with rasterio.open(os.path.join(INTERMEDIATE_PATH, f"tile.tif"), "r") as tile:
                out_image2, out_transform = mask.mask(
                    tile,
                    second_vector,
                    nodata=0,
                    all_touched=False,
                    invert=False,
                    filled=True,
                    crop=False,
                    pad=False,
                    pad_width=0.5,
                    indexes=None,
                )
            # create single band from triple band and classify binary
            out_image2 = out_image2[0]
            out_image2[out_image2 > 0] = 1

            # reduce salt-n-pepper noise
            out_image2 = median(out_image2, disk(1), mode="constant", cval=0)

            out_image3 = np.zeros(out_image.shape)
            out_image3[(out_image + out_image2) == 0] = 1
            out_image = np.array([out_image3, out_image, out_image2])  # one hot -> [Nothing, Building, Road]

        # write tile in resulting mask raster
        if os.path.exists(os.path.join(INTERMEDIATE_PATH, "masked_raster.tif")):
            with rasterio.open(
                os.path.join(INTERMEDIATE_PATH, "masked_raster.tif"),
                "r+",
                BIGTIFF="YES",
                **out_meta,
            ) as outds:
                outds.write(out_image, window=window)
        else:
            with rasterio.open(
                os.path.join(INTERMEDIATE_PATH, "masked_raster.tif"),
                "w",
                BIGTIFF="YES",
                **out_meta,
            ) as outds:
                outds.write(out_image, window=window)
    # set logging level back to info
    logger.setLevel(logging.INFO)


def create_ml_tiles(raster, r_mask) -> None:
    """
    This function breaks up the raster and mask into tiles and writes 80% into a dir for
        training, 10% into a dir for testing and 10% into a dir for validation
    :param raster: the raster data to be used for training/testing/validation
    :param r_mask: the mask which indicates the buildings
    :return: None, as data is stored on hard drive
    """
    tiles_to_be_created = math.ceil(
        (raster.meta["width"] * raster.meta["height"])
        / (TARGET_SIZE[0] * TARGET_SIZE[1])
    )
    counter_successful = 0
    for window, transform in tqdm(
        get_tiles(raster, TARGET_SIZE[0], TARGET_SIZE[1]), total=tiles_to_be_created
    ):
        if counter_successful % 10 == 0 and counter_successful != 0:
            output_path = VALIDATION_PATH
        elif counter_successful % 5 == 0 and counter_successful != 0:
            output_path = TEST_PATH
        else:
            output_path = TRAINING_PATH
        result = write_raster_window(
            raster, r_mask, window, transform, output_path, counter_successful
        )
        if result:
            counter_successful += 1
    logger.info(
        f"\n ML Data generated! {counter_successful} of possible {tiles_to_be_created} tiles were successfully created (80% train, 10% test, 10% validation)."
    )


def preprocessing_data() -> None:
    """This function runs the other functions within this script in the correct order"""
    raster = rasterio.open(RASTER_PATH)

    logger.info("Query buildings...")
    buildings = get_building_data(raster)
    logger.info(f"Number of buildings queried: {len(buildings.index)}")
    roads = None
    if DUAL_SHAPES:
        roads = get_highway_data(raster)
        logger.info(f"Number of roads queried: {len(roads.index)}")

    logger.info("Generate Mask")
    generate_mask(raster, buildings["geometry"], roads["geometry"] if DUAL_SHAPES else None)
    logger.info("Mask written")
    r_mask = rasterio.open(os.path.join(INTERMEDIATE_PATH, "masked_raster.tif"))

    logger.info("Create data for ML")

    create_ml_tiles(raster, r_mask)


if __name__ == "__main__":
    preprocessing_data()
