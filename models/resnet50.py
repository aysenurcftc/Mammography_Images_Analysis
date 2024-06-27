import ssl

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input
from tensorflow.python.keras import Sequential

import config

# Needed to download pre-trained weights for ImageNet
ssl._create_default_https_context = ssl._create_unverified_context


def create_resnet50_model(num_classes):

    img_input = Input(shape=(config.RESNET_IMG_SIZE['HEIGHT'], config.RESNET_IMG_SIZE['WIDTH'], 1))
    img_conc = Concatenate()([img_input, img_input, img_input])
    
    model_base = ResNet50(include_top=False, weights="imagenet", input_tensor=img_conc)

    # Add fully connected layers
    model = Sequential()
    model.add(model_base)
    model.add(Flatten())
    fully_connected = Sequential(name="Fully_Connected")
    fully_connected.add(Dropout(0.2, seed=config.RANDOM_SEED, name="Dropout_1"))
    fully_connected.add(Dense(units=512, activation='relu', name='Dense_1'))
    fully_connected.add(Dropout(0.2, name="Dropout_2"))
    fully_connected.add(Dense(units=32, activation='relu', name='Dense_2'))

    # Final output layer that uses softmax activation function (because the classes are exclusive).
    if num_classes == 2:
        fully_connected.add(Dense(1, activation='sigmoid', kernel_initializer="random_uniform", name='Output'))
    else:
        fully_connected.add(
            Dense(num_classes, activation='softmax', kernel_initializer="random_uniform", name='Output'))

    model.add(fully_connected)


    return model
