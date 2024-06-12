import os
from imutils import paths
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.utils import load_img, to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
import config
import tensorflow_io as tfio

def calculate_balanced_class_weights(y_train, label_encoder):
   
    if label_encoder.classes_.size != 2:
        y_train = label_encoder.inverse_transform(np.argmax(y_train, axis=1))

    # Balanced class weights
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(weights))

    if config.verbose_mode:
        print("Class weights: {}".format(str(class_weights)))

    return class_weights



def import_minimias_dataset(data_dir, label_encoder):
    
    """
    [1. 0. 0.] - This corresponds to Benign.
    [0. 1. 0.] - This corresponds to Malignant.
    [0. 0. 1.] - This corresponds to Normal.
    """
    
    # Initialize variables
    images = []
    labels = []

    # Loop over the image paths and update the data and labels lists with the preprocessed images and labels
    print("*Loading whole images*")
    for image_path in list(paths.list_images(data_dir)):
        images.append(preprocess_image(image_path))
        labels.append(image_path.split(os.path.sep)[-2])  # Extract label from path

    # Convert the data and labels lists to NumPy arrays
    images = np.array(images, dtype="float32")  # Convert images to a batch
    labels = np.array(labels)

    # Encode labels
    labels = encode_labels(labels, label_encoder)
    #for index, class_label in enumerate(label_encoder.classes_):
        #print(f"{class_label}: {index}")

    return images, labels





def import_cbisddsm_training_dataset(label_encoder):
    """
    Import the dataset getting the image paths (downloaded on BigTMP) and encoding the labels.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param label_encoder: The label encoder.
    :return: Two arrays, one for the image paths and one for the encoded labels.
    """
    print("Importing CBIS-DDSM training set")
    cbis_ddsm_path = str()
    if config.mammogram_type == "calc":
        cbis_ddsm_path = "./data/CBIS-DDSM/output/calc-test.csv"
    elif config.mammogram_type == "mass":
        cbis_ddsm_path = "./data/CBIS-DDSM/output/mass-training.csv"
    else:
        cbis_ddsm_path = "data/CBIS-DDSM/training.csv"
    df = pd.read_csv(cbis_ddsm_path)
    list_IDs = df['img_path'].values
    labels = encode_labels(df['label'].values, label_encoder)
    return list_IDs, labels


def import_cbisddsm_testing_dataset(label_encoder):
   
    print("Importing CBIS-DDSM testing set")
    cbis_ddsm_path = str()
    if config.mammogram_type == "calc":
        cbis_ddsm_path = "./data/CBIS-DDSM/output/calc-test.csv"
    elif config.mammogram_type == "mass":
        cbis_ddsm_path = "./data/CBIS-DDSM/output/mass-test.csv"
    else:
        cbis_ddsm_path = "../data/CBIS-DDSM/testing.csv"
    df = pd.read_csv(cbis_ddsm_path)
    list_IDs = df['img_path'].values
    labels = encode_labels(df['label'].values, label_encoder)
    return list_IDs, labels

def preprocess_image(image_path):
    # Only handle full images and CNN model
    if config.model == "CNN":
        target_size = (config.CNN_IMG_SIZE['HEIGHT'], config.CNN_IMG_SIZE['WIDTH'], 3)
        image = load_img(image_path, color_mode="grayscale", target_size=target_size)
    elif config.model == "MobileNet":
        target_size = (config.MOBILE_NET_IMG_SIZE['HEIGHT'], config.MOBILE_NET_IMG_SIZE["WIDTH"])
        image = load_img(image_path, color_mode="grayscale", target_size=target_size)
    else:
        target_size = (config.image_size, config.image_size) 
        image = load_img(image_path, color_mode="grayscale", target_size=target_size)
        
    
    image = img_to_array(image)
    image /= 255.0
    return image



def encode_labels(labels_list, label_encoder):
   
    labels = label_encoder.fit_transform(labels_list)
    if label_encoder.classes_.size == 2:
        return labels
    else:
        return to_categorical(labels)
    
    
    

def dataset_split(split, dataset, labels):
  
    train_X, test_X, train_Y, test_Y = train_test_split(dataset,
                                                        labels,
                                                        test_size=split,
                                                        stratify=labels,
                                                        random_state=config.RANDOM_SEED,
                                                        shuffle=True)
    return train_X, test_X, train_Y, test_Y



def create_dataset(x, y):
    """
    Generates a TF dataset for feeding in the data.
    Originally written as a group for the common pipeline.
    :param x: X inputs - paths to images
    :param y: y values - labels for images
    :return: the dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    # Map values from dicom image path to array
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Dataset to cache data and repeat until all samples have been run once in each epoch
    dataset = dataset.cache().repeat(1)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset



def parse_function(filename, label):
    image_bytes = tf.io.read_file(filename)
    image = tfio.image.decode_dicom_image(image_bytes, color_dim=True, dtype=tf.uint16)
    image = tf.image.convert_image_dtype(image[0], tf.float32)  # Convert to float32
    image = tf.image.resize(image, [config.CNN_IMG_SIZE['HEIGHT'], config.CNN_IMG_SIZE['WIDTH']])  # Resize
    image /= 255.0  # Normalize
    
    return image, label


