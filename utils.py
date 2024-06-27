from random import seed
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import config




RANDOM_SEED = 42
def set_random_seeds():
    seed(RANDOM_SEED) 
    tf.random.set_seed(RANDOM_SEED)  


def count_classes(y):
    class_counts = {}
    for label in y:
      
        label_str = str(np.argmax(label))
        if label_str in class_counts:
            class_counts[label_str] += 1
        else:
            class_counts[label_str] = 1
    return class_counts


def create_label_encoder():
    #Creates the label encoder.
    return LabelEncoder()



def gpu_available():
    
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Available GPU devices: ", physical_devices)




def load_trained_model():
    """
    Load the model previously trained for the final evaluation using the test set.
    """
    if config.model == "vit" or config.model == "ResNet":
        return load_model("/Users/aysen/OneDrive/Masaüstü/Mammography_Images_Analysis/saved_model/vision_transformer_model")
    else:
        print("Loading trained model")
        return load_model(
            "/Users/aysen/OneDrive/Masaüstü/Breast_Cancer_Detection/saved_model/dataset-{}_model-{}-lr_{}-batch_size_{}_saved-model.h5".format(
                    config.dataset,
                    config.model,
                    config.learning_rate,
                    config.batch_size,)
        )


def save_output_figure(title: str):
   
    plt.savefig(
        "./output/{}_dataset-{}_mammogramtype-{}_model-{}_{}.png".format(
            config.dataset,
            config.mammogram_type,
            config.model,
            title))  
    
