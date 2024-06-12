
# Constants
RANDOM_SEED = 42
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
DENSE_NET_IMG_SIZE = MOBILE_NET_IMG_SIZE

# models: -> CNN, MobileNet, 
#datasers -> mini-MIAS, CBIS-DDSM, 

dataset = "CBIS-DDSM"       
mammogram_type = "calc"     
model = "MobileNet"              
run_mode = "training"       
learning_rate = 0.001    
batch_size = 32  
max_epochs = 500 
is_roi = False  
max_epoch_frozen = 500      
max_epoch_unfrozen = 300    
verbose_mode = False                    
visualize = False

