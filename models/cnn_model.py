import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.metrics import CategoricalAccuracy, BinaryAccuracy
from tensorflow.python.keras.optimizers import adam_v2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import config
from models.basic_cnn import create_basic_cnn_model
from models.mobilenet_v2 import create_mobilenet_model
from visualisation.plots import plot_training_history

class CnnModel:

    def __init__(self, model_name, num_classes):
        self.model_name = model_name
        self.num_classes = num_classes
        self.history = None
        self.prediction = None



        if self.model_name == "CNN":
            self._model = create_basic_cnn_model(self.num_classes)
        elif self.model_name == "MobileNet":
            self._model = create_mobilenet_model(self.num_classes)
       
       

    def train_model(self, X_train, X_val, y_train, y_val, class_weights):
        
        if not self.model_name == "CNN":
            self.compile_model(config.learning_rate)
            self.fit_model(X_train, X_val, y_train, y_val, class_weights, is_frozen_layers=True)
        else:
            self.compile_model(config.learning_rate)
            self.fit_model(X_train, X_val, y_train, y_val, class_weights, is_frozen_layers=False)



    def compile_model(self, learning_rate):
        
         if config.dataset == "CBIS-DDSM":
            self._model.compile(optimizer=adam_v2.Adam(learning_rate),
                                loss=BinaryCrossentropy(),
                                metrics=[BinaryAccuracy()])
            
        
         elif config.dataset == "mini-MIAS":
            self._model.compile(optimizer=adam_v2.Adam(learning_rate),
                                loss=CategoricalCrossentropy(),
                                metrics=[CategoricalAccuracy()])
            
         elif config.dataset == "MBCD_Implant":
            
            if config.model == "CNN":
                self._model.compile(optimizer=Adam(learning_rate),
                                    loss=BinaryCrossentropy(),
                                    metrics=[BinaryAccuracy()])
            else:
                self._model.compile(optimizer=adam_v2.Adam(learning_rate),
                                loss=BinaryCrossentropy(),
                                metrics=[BinaryAccuracy()])
            
            
        
    def fit_model(self, X_train, X_val, y_train, y_val, class_weights, is_frozen_layers):
        
        try:
            patience = 10
            
            if config.dataset == "mini-MIAS" or config.dataset == "MBCD_Implant":
                
                self.history = self._model.fit(
                    x=X_train,
                    y=y_train,
                    class_weight=class_weights,
                    batch_size=config.batch_size,
                    validation_data=(X_val, y_val),
                    epochs=config.max_epochs,
                    verbose=1,
                    callbacks=[
                        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
                        ReduceLROnPlateau(patience=int(patience / 2))
                    ]
                )
                    
            elif config.dataset == "CBIS-DDSM":
                 self.history = self._model.fit(
                    x=X_train,
                    y=y_train,
                    validation_data=(X_val, y_val),
                    class_weight=class_weights,
                    epochs=config.max_epochs,
                    callbacks=[
                        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
                        ReduceLROnPlateau(patience=int(patience / 2))
                    ]
            )
                
        except Exception as e:
            print("Error occurred during model fitting:", str(e))
            
        plot_training_history(self.history)
            
            
    def make_prediction(self, x):
         
        try:
            if config.dataset == "mini-MIAS":
                self.prediction = self._model.predict(x=x.astype("float32"), batch_size=config.batch_size)
            elif config.dataset == "CBIS-DDSM":
                self.prediction = self._model.predict(x=x)
            elif config.dataset == "MBCD_Implant":
                self.prediction = self._model.predict(x=x)
            
        except UnicodeDecodeError as e:
            print(f"UnicodeDecodeError: {e}")
            raise e
        

    def evaluate_model(self, y_true, label_encoder, classification_type):
        
        if label_encoder.classes_.size == 2:
            y_true_inv = y_true
            y_pred_inv = np.round(self.prediction, 0)
        else:
            y_true_inv = label_encoder.inverse_transform(np.argmax(y_true, axis=1))
            y_pred_inv = label_encoder.inverse_transform(np.argmax(self.prediction, axis=1))
       

        accuracy = float('{:.3f}'.format(accuracy_score(y_true_inv, y_pred_inv)))
        precision = precision_score(y_true_inv, y_pred_inv, average='weighted')
        recall = recall_score(y_true_inv, y_pred_inv, average='weighted')
        f1 = f1_score(y_true_inv, y_pred_inv, average='weighted')

        print("Accuracy = {:.3f}".format(accuracy))
        print("Precision = {:.3f}".format(precision))
        print("Recall = {:.3f}".format(recall))
        print("F1 Score = {:.3f}".format(f1))
        
        
        
    def save_model(self):
        print("Saving model")
        if config.model == "MobileNet":
            self._model.save('/Users/aysen/OneDrive/Masaüstü/Breast_Cancer_Detection/saved_model/vision_transformer_model')
        else:
            self._model.save("/Users/aysen/OneDrive/Masaüstü/Breast_Cancer_Detection/saved_model/dataset-{}_model-{}-lr_{}-batch_size_{}_saved-model.h5".format(
                config.dataset,
                config.model,
                config.learning_rate,
                config.batch_size,)
             )
        

    def save_weights(self):
        print("Saving all weights")
        self._model.save_weights(
            f"/Users/aysen/OneDrive/Masaüstü/Breast_Cancer_Detection/saved_model/dataset-{config.dataset}_mammogramtype-{config.mammogram_type}_model-{config.model}_all_weights.h5")
        
        

def test_model_evaluation(y_true, predictions, label_encoder, classification_type):
    if label_encoder.classes_.size == 2:
        y_true_inv = y_true
        y_pred_inv = np.round(predictions, 0)
    else:
        y_true_inv = label_encoder.inverse_transform(np.argmax(y_true, axis=1))
        y_pred_inv = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
        
    accuracy = float('{:.4f}'.format(accuracy_score(y_true_inv, y_pred_inv)))
    precision = float('{:.4f}'.format(precision_score(y_true_inv, y_pred_inv, average='weighted')))
    recall = float('{:.4f}'.format(recall_score(y_true_inv, y_pred_inv, average='weighted')))
    f1 = float('{:.4f}'.format(f1_score(y_true_inv, y_pred_inv, average='weighted')))
    
    print("Accuracy = {}\n".format(accuracy))
    print("Precision = {}\n".format(precision))
    print("Recall = {}\n".format(recall))
    print("F1 Score = {}\n".format(f1))
    
    
    """ 
    if label_encoder.classes_.size == 2:  # binary classification
        plot_roc_curve_binary(y_true, predictions)
    elif label_encoder.classes_.size >= 2:  # multi classification
        plot_roc_curve_multiclass(y_true, predictions, label_encoder)
    """
  
   
  
def inverse_transform_labels(y_true, predictions, label_encoder):
    if label_encoder.classes_.size == 2:
        y_true_inv = y_true
        y_pred_inv = np.round(predictions, 0)
    else:
        y_true_inv = label_encoder.inverse_transform(np.argmax(y_true, axis=1))
        y_pred_inv = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    return y_true_inv, y_pred_inv




def calculate_accuracy(y_true, y_pred):
    return float(f'{accuracy_score(y_true, y_pred):.5f}')








