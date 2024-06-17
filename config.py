
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
#datasers -> mini-MIAS, CBIS-DDSM, MBCD_Implant

dataset = "MBCD_Implant"       
mammogram_type = "calc"     
model = "CNN"              
run_mode = "training"       
learning_rate = 0.01    
batch_size = 32
max_epochs = 100  
max_epoch_frozen = 100      
max_epoch_unfrozen = 50    
verbose_mode = True                   
visualize = False

