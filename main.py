import os
from data_operations.data_exploration import pixel_value_distribution, single_image, visualize_dataset
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import config
import tensorflow as tf
from data_operations.data_preprocessing import calculate_balanced_class_weights, create_dataset, dataset_split, import_cbisddsm_testing_dataset, import_cbisddsm_training_dataset, import_minimias_dataset
from data_operations.data_augmentation import generate_image_transforms
from models.cnn_model import CnnModel, test_model_evaluation
from utils import count_classes, create_label_encoder, gpu_available, load_trained_model, set_random_seeds


def main():
    
    gpu_available()
    set_random_seeds()
  
    # Create label encoder.
    l_e = create_label_encoder()

    # Run in training mode.
    if config.run_mode == "training":

        print("Training model\n")

        if config.dataset == "mini-MIAS":
            
            print(f"Dataset: {config.dataset}")
            print(f"Model: {config.model}")

            images, labels = import_minimias_dataset(data_dir="./data/{}/images_processed".format(config.dataset), label_encoder=l_e)
            
        
        
            # training/test/validation sets (80/20 split).
            X_train, X_test, y_train, y_test = dataset_split(split=0.20,
                                                             dataset=images,
                                                             labels=labels)




            if config.visualize:
                visualize_dataset(images, labels)
                single_image(images, labels)
                pixel_value_distribution(images, labels)
                
            
            #training/validation set (80/20 split).
            model = CnnModel(config.model, l_e.classes_.size)
            
            X_train, X_val, y_train, y_val = dataset_split(split=0.25,
                                                           dataset=X_train,
                                                           labels=y_train)
            
            
            
            
            # Calculate class weights.
            class_weights = calculate_balanced_class_weights(y_train, l_e)
            
            print("Class weights:", class_weights)
            print("X_train shape:", X_train.shape)
            print("X_val shape:", X_val.shape)
            print("y_train shape:", y_train.shape)
            print("y_val shape:", y_val.shape)

             
            # Data augmentation.
            y_train_before_data_aug = y_train
            X_train, y_train = generate_image_transforms(X_train, y_train)
            y_train_after_data_aug = y_train
            np.random.shuffle(y_train)
            
            
            if config.verbose_mode:
                before_aug_counts = count_classes(y_train_before_data_aug)
                after_aug_counts = count_classes(y_train_after_data_aug)
                
                print(f"Before data augmentation: {before_aug_counts}")
                print(f"After data augmentation: {after_aug_counts}")
                print("***********************")
                print(f"Training set size : {X_train.shape[0]}")
                print(f"Validation set size: {X_val.shape[0]}")
                print(f"Test set size: {X_test.shape[0]}")
           
              
            try:
                model.train_model(X_train, X_val, y_train, y_val, class_weights)
                
            except RuntimeError as e:
                if "Your input ran out of data" in str(e):
                    print("Training interrupted: End of sequence. Consider adjusting data or epochs.")
                else:
                    raise e  
                
                
        
        elif config.dataset == "CBIS-DDSM":
            images, labels = import_cbisddsm_training_dataset(l_e)

            # Split training dataset into training/validation sets (75%/25% split).
            X_train, X_val, y_train, y_val = dataset_split(split=0.25, dataset=images, labels=labels)
            train_dataset = create_dataset(X_train, y_train)
            validation_dataset = create_dataset(X_val, y_val)

            # Calculate class weights.
            class_weights = calculate_balanced_class_weights(y_train, l_e)

         
            model = CnnModel(config.model, l_e.classes_.size)

            # Fit model.
            if config.verbose_mode:
                print("Training set size: {}".format(X_train.shape[0]))
                print("Validation set size: {}".format(X_val.shape[0]))
            try:
                
                model.train_model(train_dataset, validation_dataset, None, None, class_weights)
                
            except RuntimeError as e:
                if "Your input ran out of data" in str(e):
                    print("Training interrupted: End of sequence. Consider adjusting data or epochs.")
                else:
                    raise e  
           
                
        # Save the model and its weights/biases.
        
        model.save_model()
        model.save_weights()


        if config.dataset == "mini-MIAS":
            model.make_prediction(X_val)
            model.evaluate_model(y_val, l_e, 'N-B-M')
            pass
        elif config.dataset == "CBIS-DDSM":
            model.make_prediction(validation_dataset)
            model.evaluate_model(y_val, l_e, 'B-M')
            
        
      
      

    # Run in testing mode.
    elif config.run_mode == "test":

        print("**** Testing model ****\n")

    
        #mini-MIAS dataset
        if config.dataset == "mini-MIAS":
            images, labels = import_minimias_dataset(data_dir="./data/{}/images_processed".format(config.dataset),
                                                     label_encoder=l_e)
            
            _, X_test, _, y_test = dataset_split(split=0.20, dataset=images, labels=labels)
            model = load_trained_model()
            predictions = model.predict(x=X_test)
            test_model_evaluation(y_test, predictions, l_e, 'N-B-M')
            
            
        # Test binary classification (CBIS-DDSM dataset).
        elif config.dataset == "CBIS-DDSM":
            images, labels = import_cbisddsm_testing_dataset(l_e)
            test_dataset = create_dataset(images, labels)
            model = load_trained_model()
            predictions = model.predict(x=test_dataset)
            test_model_evaluation(labels, predictions, l_e, 'B-M')
            
    
    
    
if __name__ == '__main__':
    main()
