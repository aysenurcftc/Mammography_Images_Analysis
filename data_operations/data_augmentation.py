import random
import cv2
import numpy as np
import skimage as sk
import skimage.transform

import config


def generate_image_transforms(images, labels):
   
    augmentation_multiplier = 3

    images_with_transforms = images
    labels_with_transforms = labels

    available_transforms = {
        'rotate': random_rotation,
        'noise': random_noise,
        'horizontal_flip': horizontal_flip,
        'shear': random_shearing,
        'blur': filters_blur,
        'gaussianblur': filters_gaussianblur,
    
    }

    
    class_balance = get_class_balances(labels)
    max_count = max(class_balance) * augmentation_multiplier  
    to_add = [max_count - i for i in class_balance]

    for i in range(len(to_add)):
        if int(to_add[i]) == 0:
            continue
        label = np.zeros(len(to_add))
        label[i] = 1
        indices = [j for j, x in enumerate(labels) if np.array_equal(x, label)]
        indiv_class_images = [images[j] for j in indices]

        for k in range(int(to_add[i])):
            a = create_individual_transform(indiv_class_images[k % len(indiv_class_images)], available_transforms)
            transformed_image = create_individual_transform(indiv_class_images[k % len(indiv_class_images)],
                                                            available_transforms)

            if config.model == "CNN":
                transformed_image = transformed_image.reshape(1, config.ROI_IMG_SIZE['HEIGHT'],
                                                              config.ROI_IMG_SIZE["WIDTH"], 1)
         
            elif config.model == "MobileNet":
                transformed_image = transformed_image.reshape(1, config.MOBILE_NET_IMG_SIZE['HEIGHT'],
                                                              config.MOBILE_NET_IMG_SIZE["WIDTH"], 1)
                
            elif config.model == "vit":
                transformed_image = transformed_image.reshape(1, config.MOBILE_NET_IMG_SIZE['HEIGHT'],
                                                              config.MOBILE_NET_IMG_SIZE["WIDTH"], 1)
         
            images_with_transforms = np.append(images_with_transforms, transformed_image, axis=0)
            
            transformed_label = label.reshape(1, len(label))
            labels_with_transforms = np.append(labels_with_transforms, transformed_label, axis=0)


    return images_with_transforms, labels_with_transforms


def random_rotation(image_array: np.ndarray):
  
    random_degree = random.uniform(-20, 20)
    return sk.transform.rotate(image_array, random_degree)


def random_noise(image):
   
    return sk.util.random_noise(image)


def filters_blur(image):
    fsize = 9
    return cv2.blur(image,(fsize,fsize))


def filters_gaussianblur(image):
    fsize = 9
    return cv2.GaussianBlur(image, (fsize, fsize), 0)



def filters_medianblur(image):
    fsize = 9
    return cv2.medianBlur(image, fsize)

def horizontal_flip(image):
   
    return image[:, ::-1]




def random_shearing(image):
   
    random_degree = random.uniform(-0.2, 0.2)
    tf = sk.transform.AffineTransform(shear=random_degree)
    return sk.transform.warp(image, tf, order=1, preserve_range=True, mode='wrap')


def create_individual_transform(image, transforms):
   
    num_transformations_to_apply = random.randint(1, len(transforms))
    num_transforms = 0
    transformed_image = None
    while num_transforms <= num_transformations_to_apply:
        key = random.choice(list(transforms))
        transformed_image = transforms[key](image)
        num_transforms += 1

    return transformed_image


def get_class_balances(y_vals):
   
    if config.dataset == "mini-MIAS":
        num_classes = len(y_vals[0])
        counts = np.zeros(num_classes)
        for y_val in y_vals:
            for i in range(num_classes):
                counts[i] += y_val[i]
    return counts.tolist()
