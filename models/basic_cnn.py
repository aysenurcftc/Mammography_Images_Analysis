from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
import config


def create_basic_cnn_model(num_classes: int):
    
    inputs = Input(shape=(config.ROI_IMG_SIZE['HEIGHT'], config.ROI_IMG_SIZE['WIDTH'], 1))

    # Convolutional layers with batch normalization
    conv1 = Conv2D(16, (3, 3), activation='relu')(inputs)
    batch_norm1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(batch_norm1)

    conv2 = Conv2D(32, (3, 3), activation='relu')(pool1)
    batch_norm2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(batch_norm2)

   
    flat = Flatten()(pool2)

    # Fully connected (dense) layers
    dense1 = Dense(64, activation='relu')(flat)

    # Output layer
    if num_classes == 2:
        outputs = Dense(1, activation='sigmoid')(dense1)
    else:
        outputs = Dense(num_classes, activation='softmax')(dense1)

    # Model definition
    model = Model(inputs=inputs, outputs=outputs)
    
     # Print model details if running in debug mode.
    if config.verbose_mode:
        print(model.summary())

    return model