import numpy as np
import os
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
from tensorflow import keras
import tensorflow_addons as tfa
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from models.basic_cnn import create_basic_cnn_model
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
from models.mobilenet_v2 import create_mobilenet_model
from models.resnet50 import create_resnet50_model
from models.vit import MultiHeadAttentionLSA, PatchEncoder, ShiftedPatchTokenization
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
        elif self.model_name == "vit":
            self._model = self.create_vit_classifier()
        elif self.model_name == "ResNet":
            self._model = create_resnet50_model(self.num_classes)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    

    def train_model(self, X_train, X_val, y_train, y_val, class_weights):
        
        
          if not self.model_name == "CNN":

            if self.model_name == "ResNet" or self.model_name == "MobileNet":
                self._model.layers[0].trainable = False
              
            self.compile_model(config.learning_rate)
            self.fit_model(X_train, X_val, y_train, y_val, class_weights, is_frozen_layers=True)

         
            if self.model_name == "ResNet" or self.model_name == "MobileNet":
                self._model.layers[0].trainable = True
           

            
            self.compile_model(1e-5)  # Very low learning rate.
            self.fit_model(X_train, X_val, y_train, y_val, class_weights, is_frozen_layers=False)


          else:
            self.compile_model(config.learning_rate)
            self.fit_model(X_train, X_val, y_train, y_val, class_weights, is_frozen_layers=True)
        

    def compile_model(self, learning_rate):
        
         if config.dataset == "CBIS-DDSM":
             
            if config.model == "vit":
                optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=config.WEIGHT_DECAY)

                if config.NUM_CLASSES == 2:
                    loss = keras.losses.BinaryCrossentropy(from_logits=False)  # from_logits=False because we use sigmoid activation
                    metrics = [keras.metrics.BinaryAccuracy(name="accuracy")]
                else:
                    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)  # from_logits=True because last layer has no activation
                    metrics = [
                        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),  
                    ]

                self._model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=metrics
                )
            else:
                self._model.compile(optimizer=Adam(learning_rate),
                                    loss=BinaryCrossentropy(),
                                    metrics=[BinaryAccuracy()])
            
        
         elif config.dataset == "mini-MIAS":
             
            if config.model == "vit":
                optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=config.WEIGHT_DECAY)

                if config.NUM_CLASSES == 2:
                    loss = keras.losses.BinaryCrossentropy(from_logits=False)  # from_logits=False because we use sigmoid activation
                    metrics = [keras.metrics.BinaryAccuracy(name="accuracy")]
                else:
                    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # from_logits=True because last layer has no activation
                    metrics = [
                        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                        keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
                        
                    ]

                self._model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=metrics
                )
                
            else:
                self._model.compile(optimizer=Adam(learning_rate),
                                    loss=CategoricalCrossentropy(),
                                    metrics=[CategoricalAccuracy()])
                
            
            
         elif config.dataset == "MBCD_Implant":
            
            if config.model == "CNN":
                self._model.compile(optimizer=Adam(learning_rate),
                                    loss=BinaryCrossentropy(),
                                    metrics=[BinaryAccuracy()])
                
            elif config.model == "vit":
                optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=config.WEIGHT_DECAY)

                if config.NUM_CLASSES == 2:
                    loss = keras.losses.BinaryCrossentropy(from_logits=False)  # from_logits=False because we use sigmoid activation
                    metrics = [keras.metrics.BinaryAccuracy(name="accuracy")]
                else:
                    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)  # from_logits=True because last layer has no activation
                    metrics = [
                        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),  
                    ]

                self._model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=metrics
                )
                   
            else:
                self._model.compile(optimizer=adam_v2.Adam(learning_rate),
                                loss=BinaryCrossentropy(),
                                metrics=[BinaryAccuracy()])
                
         elif config.dataset == "CLAHE_images":
             if config.model == "vit":
                optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=config.WEIGHT_DECAY)

                if config.NUM_CLASSES == 2:
                    loss = keras.losses.BinaryCrossentropy(from_logits=False)  # from_logits=False because we use sigmoid activation
                    metrics = [keras.metrics.BinaryAccuracy(name="accuracy")]
                else:
                    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)  # from_logits=True because last layer has no activation
                    metrics = [
                        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),  
                    ]

                self._model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=metrics
                )
             else:
                self._model.compile(optimizer=Adam(learning_rate),
                                    loss=BinaryCrossentropy(),
                                    metrics=[BinaryAccuracy()])
            
            
        
    def fit_model(self, X_train, X_val, y_train, y_val, class_weights, is_frozen_layers):
        
        try:
            
            if is_frozen_layers:
                max_epochs = config.max_epoch_frozen
                patience = 10
            else:
                max_epochs = config.max_epoch_unfrozen
                patience = 10
                
            if config.dataset == "mini-MIAS":
                self.history = self._model.fit(
                    x=X_train,
                    y=y_train,
                    class_weight=class_weights,
                    batch_size=config.batch_size,
                    steps_per_epoch=len(X_train) // config.batch_size,
                    validation_data=(X_val, y_val),
                    validation_steps=len(X_val) // config.batch_size,
                    epochs=max_epochs,
                    callbacks=[
                        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
                        ReduceLROnPlateau(patience=int(patience / 2))
                    ]
                )
           
            elif config.dataset == "CBIS-DDSM":
                self.history = self._model.fit(
                    x=X_train,
                    validation_data=X_val,
                    class_weight=class_weights,
                    epochs=max_epochs,
                    callbacks=[
                        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
                        ReduceLROnPlateau(patience=int(patience / 2))
                    ]
                )
                
            else:
                self.history = self._model.fit(
                    x=X_train,
                    y=y_train,
                    batch_size=config.batch_size,
                    steps_per_epoch=len(X_train) // config.batch_size,
                    validation_data=(X_val, y_val),
                    validation_steps=len(X_val) // config.batch_size,
                    epochs=max_epochs,
                    callbacks=[
                        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
                        ReduceLROnPlateau(patience=int(patience / 2))
                    ]
                )
                
                        
        except Exception as e:
            print("Error occurred during model fitting:", str(e))
            
        #plot_training_history(self.history)
            
            
    def make_prediction(self, x):
        try:
            if config.dataset == "mini-MIAS":
                self.prediction = self._model.predict(x=x.astype("float32"), batch_size=config.batch_size)
               
            elif config.dataset == "CBIS-DDSM":
                self.prediction = self._model.predict(x=x)
            elif config.dataset == "MBCD_Implant":
                self.prediction = self._model.predict(x=x)
                
            elif config.dataset == "CLAHE_images":
                self.prediction = self._model.predict(x=x)
            
        except UnicodeDecodeError as e:
            print(f"UnicodeDecodeError: {e}")
            raise e
        

    def evaluate_model(self, y_true, label_encoder, classification_type=None):
        
        if label_encoder.classes_.size == 2:
            
            y_true_inv = y_true
            y_pred_inv = np.round(self.prediction, 0).astype(int)
        else:
            y_true_inv = label_encoder.inverse_transform(np.argmax(y_true, axis=1))
            y_pred_inv = label_encoder.inverse_transform(np.argmax(self.prediction, axis=1))
            
            print(len(y_true_inv))
            print(len(y_pred_inv))
       
       
        print("Shape of y_true:", y_true.shape)
        print("Unique values in y_true:", np.unique(y_true))
        print("Shape of predictions:", self.prediction.shape)
        print("Unique values in predictions:", np.unique(np.round(self.prediction, 0).astype(int)))

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
        if config.model == "MobileNet" or config.model == "vit" or config.model == "ResNet":
            self._model.save('/Users/aysen/OneDrive/Masaüstü/Mammography_Images_Analysis/saved_model/vision_transformer_model',save_format='tf')
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
        
     
        """ 
        if label_encoder.classes_.size == 2:  # binary classification
            plot_roc_curve_binary(y_true, predictions)
        elif label_encoder.classes_.size >= 2:  # multi classification
            plot_roc_curve_multiclass(y_true, predictions, label_encoder)
        """
    
    
    
    def inverse_transform_labels(self, y_true, predictions, label_encoder):
        if label_encoder.classes_.size == 2:
            y_true_inv = y_true
            y_pred_inv = np.round(predictions, 0).astype(int)
        else:
            y_true_inv = label_encoder.inverse_transform(np.argmax(y_true, axis=1))
            y_pred_inv = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
        return y_true_inv, y_pred_inv




    def calculate_accuracy(self, y_true, y_pred):
        return float(f'{accuracy_score(y_true, y_pred):.5f}')


    def create_vit_classifier(self):
        inputs = layers.Input(shape=config.INPUT_SHAPE)
        (tokens, _) = ShiftedPatchTokenization()(inputs)
        encoded_patches = PatchEncoder()(tokens)

        for _ in range(config.TRANSFORMER_LAYERS):
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = MultiHeadAttentionLSA(
                num_heads=config.NUM_HEADS, key_dim=config.PROJECTION_DIM, dropout=0.1
            )(x1, x1)
            x2 = layers.Add()([attention_output, encoded_patches])
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = self.mlp(x3, hidden_units=config.TRANSFORMER_UNITS, dropout_rate=0.1)
            encoded_patches = layers.Add()([x3, x2])

        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        features = self.mlp(representation, hidden_units=config.MLP_HEAD_UNITS, dropout_rate=0.5)
        if config.NUM_CLASSES == 2:
            logits = layers.Dense(1, activation='sigmoid')(features)
        else:
            logits = layers.Dense(3)(features)
        
        model = keras.Model(inputs=inputs, outputs=logits)
        return model
        
        
        
    def mlp(self, x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x





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