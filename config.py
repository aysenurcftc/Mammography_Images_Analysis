
# Constants
RANDOM_SEED = 42

#lr -> 0.00001, 0.000001, 

# models: -> CNN, vit
#datasets -> mini-MIAS, CBIS-DDSM, MBCD_Implant, CLAHE_images

dataset = "CBIS-DDSM"       
mammogram_type ="calc"     
model = "vit"              
run_mode ="training"       
learning_rate = 0.000001    
batch_size = 8
max_epoch = 100
verbose_mode = True                   
visualize = False



MINI_MIAS_IMG_SIZE = {
    "HEIGHT": 1024,
    "WIDTH": 1024
}


ROI_IMG_SIZE = {
    "HEIGHT": 224,
    "WIDTH": 224
}

CNN_IMG_SIZE = {
     "HEIGHT": 224,
    "WIDTH": 224
}

MOBILE_NET_IMG_SIZE = {
    "HEIGHT": 224,
    "WIDTH": 224
}


RESNET_IMG_SIZE = MOBILE_NET_IMG_SIZE

#******************** VİT ***************************#
# DATA
BUFFER_SIZE = 512
BATCH_SIZE = 256

# AUGMENTATION
image_size= 224
IMAGE_SIZE = 72
PATCH_SIZE = 6
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

# OPTIMIZER
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

# TRAINING
EPOCHS = 50

# ARCHITECTURE
LAYER_NORM_EPS = 1e-6
TRANSFORMER_LAYERS = 8
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_UNITS = [
    PROJECTION_DIM * 2,
    PROJECTION_DIM,
]
MLP_HEAD_UNITS = [2048, 1024]

NUM_CLASSES = 2
INPUT_SHAPE = (224, 224, 1)
#****************************************************




