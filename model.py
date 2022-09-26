"""
This Script defines the Image generators and builds the model used by unet.py
"""
import os

from tensorflow.keras import Model, layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from definitions import DATA_PATH, TRAINING_PATH
from main import DUAL_SHAPES


## generator
def get_generator(batch_size, target_size, mode):
    """this function generates the different image data generators"""
    seed = 42
    gen_train_img = ImageDataGenerator(
        # rescales and augments data
        rescale=1.0 / 255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    gen_train_mask = ImageDataGenerator(
        # augments data
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    train_generator_img = gen_train_img.flow_from_directory(
        # defines generators data source
        TRAINING_PATH,
        classes=["img"],
        batch_size=batch_size,
        class_mode=None,
        target_size=target_size,
        seed=seed,
        shuffle=True,
    )
    train_generator_mask = gen_train_mask.flow_from_directory(
        # defines generators data source
        TRAINING_PATH,
        classes=["mask"],
        batch_size=batch_size,
        class_mode=None,
        target_size=target_size,
        seed=seed,
        shuffle=True,
        color_mode="grayscale" if not DUAL_SHAPES else "rgb",
    )
    no_of_trainsets = train_generator_img.samples
    TRAIN_GENERATOR = zip(
        train_generator_img, train_generator_mask
    )  # combine into one to yield both at the same time
    PATH = os.path.join(DATA_PATH, mode)
    # test generator only rescales, but does not augment data
    gen_test_img = ImageDataGenerator(rescale=1.0 / 255.0)
    gen_test_mask = ImageDataGenerator()
    test_generator_img = gen_test_img.flow_from_directory(
        PATH,
        classes=["img"],
        batch_size=batch_size,
        class_mode=None,
        target_size=target_size,
        seed=seed,
        shuffle=False,
    )
    test_generator_mask = gen_test_mask.flow_from_directory(
        PATH,
        classes=["mask"],
        batch_size=batch_size,
        class_mode=None,
        target_size=target_size,
        seed=seed,
        shuffle=False,
        color_mode="grayscale" if not DUAL_SHAPES else "rgb",
    )
    no_of_testsets = test_generator_img.samples
    TEST_GENERATOR = zip(test_generator_img, test_generator_mask)
    return TRAIN_GENERATOR, TEST_GENERATOR, no_of_trainsets, no_of_testsets


## cnn
def conv_block(input, num_filters):
    """In this function, a convolutional block is defined"""
    x = layers.Conv2D(num_filters, 3, padding="same")(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def decoder_block(input, skip_features, num_filters, no_of_conv_blocks):
    """In this function, the decoder blocks are defined by adding Transpose- and
    concatenate layers and convolutional blocks together"""
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = layers.Concatenate()([x, skip_features])
    for _ in range(no_of_conv_blocks):
        x = conv_block(x, num_filters)
    return x


def build_model(target_size):
    """Build the unet."""
    inputs = layers.Input(shape=target_size + [3])

    # get pretrained model
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)

    # make the first pretrained layer untrainable
    for layer in vgg16.layers[:-8]:
        layer.trainable = False

    # encoder layer
    e1 = vgg16.get_layer("block1_conv2").output
    e2 = vgg16.get_layer("block2_conv2").output
    e3 = vgg16.get_layer("block3_conv3").output
    e4 = vgg16.get_layer("block4_conv3").output
    e5 = vgg16.get_layer("block5_conv3").output

    # bottom layer
    last_pool = vgg16.get_layer("block5_pool").output

    # decoder layer
    d1 = decoder_block(last_pool, e5, 512, 3)
    d2 = decoder_block(d1, e4, 512, 3)
    d3 = decoder_block(d2, e3, 256, 3)
    d4 = decoder_block(d3, e2, 128, 2)
    d5 = decoder_block(d4, e1, 64, 2)

    # output
    if DUAL_SHAPES:
        outputs = layers.Conv2D(3, 1, padding="same", activation="sigmoid")(d5)

    else:
        outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d5)

    # initiating model
    model = Model(inputs, outputs, name="infra_cnn")

    # compile model
    model.compile(loss=BinaryCrossentropy() if not DUAL_SHAPES else CategoricalCrossentropy(), optimizer=Nadam(), metrics=["accuracy"])

    return model
