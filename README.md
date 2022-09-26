# Infrastructure CNN

This repository was created for a machine learning course at University Heidelberg. It tries to create a raster which contains all building and road footprints based on a training with osm building and road data in the same region.

There are 4 important scripts. The CNN and Image generators are defined in model.py, the preprocessing.py turns an rgb raster into the training data, and unet.py defines training, predicting and evaluation.

The scripts are run through the main.py, at the top you can define some hyperparameters for preprocessing and training and at the bottom you can choose if you want to run preprocessing, training or both.

It is **required** to have a folder named "data" at the highest level with a raster that should be classified, it should be called *aoi.tif*, but that name can be changed in the script definitions.