import os
import argparse
from data_operations.data_exploration import pixel_value_distribution, single_image, visualize_dataset
from models.cnn_model import CnnModel, test_model_evaluation
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import config
import tensorflow as tf
from data_operations.data_preprocessing import calculate_balanced_class_weights, create_dataset, dataset_split, import_cbisddsm_testing_dataset, import_cbisddsm_training_dataset, import_dataset
from data_operations.data_augmentation import generate_image_transforms
from utils import count_classes, create_label_encoder, gpu_available, load_trained_model, set_random_seeds


def main(args):
    
    gpu_available()
    set_random_seeds()
  
    # Create label encoder.
    l_e = create_label_encoder()

    # Run in training mode.
    if args.run_mode == "training":

        if args.dataset == "CBIS-DDSM":
            
            images, labels = import_cbisddsm_training_dataset(l_e)

            # Split training dataset into training/validation sets (75%/25% split).
            X_train, X_val, y_train, y_val = dataset_split(split=0.25, dataset=images, labels=labels)
            train_dataset = create_dataset(X_train, y_train)
            validation_dataset = create_dataset(X_val, y_val)
            
           
            # Calculate class weights.
            class_weights = calculate_balanced_class_weights(y_train, l_e)

         
            model = CnnModel(args.model, l_e.classes_.size)



            if args.visualize:
                visualize_dataset(images, labels)
                single_image(images, labels)
                pixel_value_distribution(images, labels)
                
                
            # Fit model.
            if args.verbose_mode:
                print(f"Training set size: {X_train.shape[0]}")
                print(f"Validation set size: {X_val.shape[0]}")
            try:
                
                model.train_model(train_dataset, validation_dataset, None, None, class_weights)
                
            except RuntimeError as e:
                if "Your input ran out of data" in str(e):
                    print("Training interrupted: End of sequence. Consider adjusting data or epochs.")
                else:
                    raise e  
                
          
        elif args.dataset == "CLAHE_images":
            data_dir = "./data/CLAHE_images"      
            images, labels = import_dataset(data_dir, l_e)
            
             # training/test/validation sets (80/20 split).
            X_train, X_test, y_train, y_test = dataset_split(split=0.20,
                                                             dataset=images,
                                                             labels=labels)

            #training/validation set (80/20 split).
            model = CnnModel(args.model, l_e.classes_.size)
            
            X_train, X_val, y_train, y_val = dataset_split(split=0.25,
                                                           dataset=X_train,
                                                           labels=y_train)
            # Calculate class weights.
            class_weights = calculate_balanced_class_weights(y_train, l_e)
            
            print("X_train shape:", X_train.shape)
            print("X_val shape:", X_val.shape)
            print("y_train shape:", y_train.shape)
            print("y_val shape:", y_val.shape)

             
            try:
                model.train_model(X_train, X_val, y_train, y_val, class_weights)
                
            except RuntimeError as e:
                if "Your input ran out of data" in str(e):
                    print("Training interrupted: End of sequence. Consider adjusting data or epochs.")
                else:
                    raise e  
            
                
        elif args.dataset == "MBCD_Implant":
            
            data_dir = "./data/MBCD_Implant/train"
            
            images, labels = import_dataset(data_dir, l_e)
            
             # training/test/validation sets (80/20 split).
            X_train, X_test, y_train, y_test = dataset_split(split=0.20,
                                                             dataset=images,
                                                             labels=labels)

            #training/validation set (80/20 split).
            model = CnnModel(args.model, l_e.classes_.size)
            
            X_train, X_val, y_train, y_val = dataset_split(split=0.25,
                                                           dataset=X_train,
                                                           labels=y_train)
            # Calculate class weights.
            class_weights = calculate_balanced_class_weights(y_train, l_e)
            
            print("X_train shape:", X_train.shape)
            print("X_val shape:", X_val.shape)
            print("y_train shape:", y_train.shape)
            print("y_val shape:", y_val.shape)

             
            try:
                model.train_model(X_train, X_val, y_train, y_val, class_weights)
                
            except RuntimeError as e:
                if "Your input ran out of data" in str(e):
                    print("Training interrupted: End of sequence. Consider adjusting data or epochs.")
                else:
                    raise e  
            
              
        #Save the model and its weights/biases.
        model.save_model()
        model.save_weights()


        if args.dataset == "CBIS-DDSM":
            model.make_prediction(validation_dataset)
            model.evaluate_model(y_val, l_e, 'B-M')
            
        elif args.dataset == "MBCD_Implant":
            model.make_prediction(X_test)
            model.evaluate_model(y_test, l_e, classification_type="binary")
            
        elif args.dataset == "CLAHE_images":
            model.make_prediction(X_test)
            model.evaluate_model(y_test, l_e, 'N-B-M')
            
            
        
      
    # Run in testing mode.
    elif args.run_mode == "test":

        print("**** Testing model ****\n")

        # Test binary classification (CBIS-DDSM dataset).
        if args.dataset == "CBIS-DDSM":
            images, labels = import_cbisddsm_testing_dataset(l_e)
            test_dataset = create_dataset(images, labels)
            model = load_trained_model()
            predictions = model.predict(x=test_dataset)
            test_model_evaluation(labels, predictions, l_e, 'B-M')
            
            
            
        elif args.dataset == "MBCD_Implant":
            
            data_dir = "./data/MBCD_Implant"
            
            images, labels = import_dataset(data_dir, l_e)
            
            _, X_test, _, y_test = dataset_split(split=0.20, dataset=images, labels=labels)
            
            model = load_trained_model()
            
            predictions = model.predict(x=X_test)
            test_model_evaluation(y_test, predictions, l_e, classification_type="binary")
            
            
        elif args.dataset == "CLAHE_images":
            
            data_dir = "./data/CLAHE_images"
            
            images, labels = import_dataset(data_dir, l_e)
            
            _, X_test, _, y_test = dataset_split(split=0.20, dataset=images, labels=labels)
            
            model = load_trained_model()
            
            predictions = model.predict(x=X_test)
            test_model_evaluation(y_test, predictions, l_e, classification_type="binary")
            
            
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training and testing a CNN model for different datasets.")
    
    parser.add_argument("--dataset", type=str, default="CBIS-DDSM", choices=["CBIS-DDSM", "CLAHE_images", "MBCD_Implant"], help="Dataset to be used")
    parser.add_argument("--mammogram_type", type=str, default="calc", help="Type of mammogram")
    parser.add_argument("--model", type=str, default="vit", choices=["cnn", "vit"], help="Model to be used")
    parser.add_argument("--run_mode", type=str, default="training", choices=["training", "test"], help="Run mode: training or test")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--verbose_mode", type=bool, default=True, help="Verbose mode")
    parser.add_argument("--visualize", type=bool, default=False, help="Visualize dataset")
    args = parser.parse_args()
    
    main(args)
