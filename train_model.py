import os
import configparser
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from app import API
# from werkzeug.serving import is_running_from_reloader
import shutil

class TrainModel:
    def __init__(self):
        self.path_train_dataset = os.path.join(os.getcwd(), 'dataset', 'train')
        self.path_test_dataset = os.path.join(os.getcwd(), 'dataset', 'test')

        # Set config.ini values
        settings_config = configparser.ConfigParser()
        settings_config.read_file(open(r"config.ini", encoding="utf-8"))
        self.target_size = int(settings_config["DATASET_PARAMS"]["target_size"])
        self.batch_size = int(settings_config["DATASET_PARAMS"]["batch_size"])
        self.epochs = int(settings_config["TRAIN_PARAMS"]["epochs"])
        self.patience = int(settings_config["TRAIN_PARAMS"]["patience"])





    def setup_data_augmentation(self):
        """
        Get params in arg if any and set them accordingly below
        -- if no params req from user, have this merged into setup_dataset()
        :return:
        """
        # Data Augmentation setup
        datagen = ImageDataGenerator(
            rescale=1. / 255,  # Normalize pixel values
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        return datagen

    def load_dataset(self, datagen):
        """
        Loads Training and Validation dataset
        :param datagen:
        :return:
        """
        # Load training and testing data with ImageDataGenerator
        train_gen = datagen.flow_from_directory(
            self.path_train_dataset,
            target_size=(self.target_size, self.target_size),  # Resize images
            batch_size=self.batch_size,
            class_mode='binary',  # Binary classification: cataract vs normal
            shuffle=True
        )

        # Validation data (you can use the same validation set or create a separate validation set)
        val_gen = datagen.flow_from_directory(
            self.path_test_dataset,
            target_size=(self.target_size, self.target_size),
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )

        return train_gen, val_gen

    def build_model(self, input_shape=(128, 128, 3)):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.target_size, self.target_size, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Binary classification (0 or 1)
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, model, train_gen, val_gen):
        # Define callbacks
        log_dir = 'logs/'  # Directory where TensorBoard logs will be stored

        callbacks = [
            EarlyStopping(patience=self.patience, restore_best_weights=True),
            TensorBoard(log_dir=log_dir, histogram_freq=1),  # Log training data for TensorBoard
            # ModelCheckpoint('best_model.h5', save_best_only=True)
        ]

        history = model.fit(
            train_gen,
            epochs=self.epochs,
            validation_data=val_gen,
            callbacks=callbacks
        )

        # Save the trained model
        model.save('best_model.h5')

        return history

    def plot_graphs(self, history, model, val_gen):
        # Save training and validation loss over epochs

        if os.path.exists("output"):
            shutil.rmtree("output")  
        os.mkdir("output")
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.savefig(os.path.join(os.getcwd(), "output", "train_val_loss_curve.png"))
        plt.close() 

        val_preds = model.predict(val_gen)
        val_preds = (val_preds > 0.5).astype(int)

        cm = confusion_matrix(val_gen.classes, val_preds)
        print(cm)

        # Calculate F1-Score
        f1 = f1_score(val_gen.classes, val_preds)

        # Create a heatmap for the confusion matrix
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cataract', 'Normal'], yticklabels=['Cataract', 'Normal'])

        plt.title(f'Confusion Matrix (F1-Score: {f1:.2f})')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Save the confusion matrix plot to an image file
        plt.savefig(os.path.join(os.getcwd(), "output", "confusion_matrix.png"))
        plt.close()  # Close the plot to avoid display

        # Calculate ROC curve and AUC score
        fpr, tpr, thresholds = roc_curve(val_gen.classes, val_preds)
        auc = roc_auc_score(val_gen.classes, val_preds)

        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')

        # Add title and labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (AUC = {auc:.2f})')

        # Save the ROC curve plot to an image file
        plt.savefig(os.path.join(os.getcwd(), "output", "roc_curve.png"))
        plt.close()  # Close the plot to avoid display


