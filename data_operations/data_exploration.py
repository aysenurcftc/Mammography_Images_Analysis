# Import necessary packages
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config
from data_operations.data_preprocessing import import_minimias_dataset



def read_csv(datadir):
    # Read csv file containing training datadata
    train_df = pd.read_csv(datadir)
    
    return train_df


def visualize_dataset(data, label):
    
    if config.dataset == "mini-MIAS":
        fig = plt.figure(figsize=(15, 10))
        for i in range(6):
            rand = random.randint(0, len(label) - 1)
            ax = plt.subplot(2, 3, i + 1)
        
            img = data[rand]  
            img = cv2.resize(img, (256, 256)) 
            plt.imshow(img, cmap='gray')
            
            if np.array_equal(label[rand], [1, 0, 0]): # [1. 0. 0.] - This corresponds to Benign
                plt.title("Benign")
            elif np.array_equal(label[rand], [0, 1, 0]): # [0. 1. 0.] - This corresponds to Malignant.
                plt.title("Malignant")
            else:
                plt.title("Normal") # [0. 0. 1.] - This corresponds to Normal.

            plt.axis('off')
                
        plt.show()
            
           
def  single_image(data, label):
    
    if config.dataset=="mini-MIAS":
        fig = plt.figure(figsize=(15, 10))
        rand = random.randint(0, len(label) - 1)
        img = data[rand]  
        img = cv2.resize(img, (256, 256)) 
        plt.imshow(img, cmap='gray')
        plt.colorbar()
        if np.array_equal(label[rand], [1, 0, 0]): # [1. 0. 0.] - This corresponds to Benign
            plt.title("Benign")
        elif np.array_equal(label[rand], [0, 1, 0]): # [0. 1. 0.] - This corresponds to Malignant.
            plt.title("Malignant")
        else:
            plt.title("Normal") # [0. 0. 1.] - This corresponds to Normal.
        plt.show()
        
        print(f"The dimensions of the image are {img.shape[0]} pixels width and {img.shape[1]} pixels height, one single color channel")
        print(f"The maximum pixel value is {img.max():.4f} and the minimum is {img.min():.4f}")
        print(f"The mean value of the pixels is {img.mean():.4f} and the standard deviation is {img.std():.4f}")
        
        
    

def pixel_value_distribution(data, label):
    
    rand = random.randint(0, len(label) - 1)
    image = data[rand]  
    
    # Plot a histogram of the distribution of the pixels
    sns.histplot(image.ravel(), 
                label=f'Pixel Mean {np.mean(image):.4f} & Standard Deviation {np.std(image):.4f}', kde=True)
    plt.legend(loc='upper center')
    plt.title('Distribution of Pixel Intensities in the Image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('# Pixels in Image')
    plt.show()
    

