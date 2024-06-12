import ssl

from keras.applications import MobileNetV2
from keras.layers import Concatenate, Dense, Dropout, Flatten, Input
from tensorflow.python.keras import Sequential

import config

# Needed to download pre-trained weights for ImageNet
ssl._create_default_https_context = ssl._create_unverified_context


def create_mobilenet_model(num_classes: int):
    
    img_input = Input(shape=(config.DENSE_NET_IMG_SIZE['HEIGHT'], config.DENSE_NET_IMG_SIZE['WIDTH'], 1))
    img_conc = Concatenate()([img_input, img_input, img_input])

    # Generate a MobileNetV2 model with pre-trained ImageNet weights, input as given above, excluded fully connected layers.
    model_base = MobileNetV2(include_top=False, weights="imagenet", input_tensor=img_conc)

    # Add fully connected layers
    model = Sequential()
    # Start with base model consisting of convolutional layers
    model.add(model_base)

    model.add(Flatten())

    fully_connected = Sequential(name="Fully_Connected")
    # Fully connected layers.
    fully_connected.add(Dropout(0.2, seed=config.RANDOM_SEED, name="Dropout_1"))
    fully_connected.add(Dense(units=512, activation='relu', name='Dense_1'))
    # fully_connected.add(Dropout(0.2, name="Dropout_2"))
    fully_connected.add(Dense(units=32, activation='relu', name='Dense_2'))

    # Final output layer that uses softmax activation function (because the classes are exclusive).
    if num_classes == 2:
        fully_connected.add(Dense(1, activation='sigmoid', kernel_initializer="random_uniform", name='Output'))
    else:
        fully_connected.add(
            Dense(num_classes, activation='softmax', kernel_initializer="random_uniform", name='Output'))

    model.add(fully_connected)

    # Print model details if running in debug mode.
    if config.verbose_mode:
        print("CNN Model used:")
        print(model.summary())
        print("Fully connected layers:")
        print(fully_connected.summary())

    return model
