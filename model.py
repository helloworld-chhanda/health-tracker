import numpy as np
import pandas as pd
import warnings
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import json
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pyngrok import ngrok

# Suppress warnings
warnings.filterwarnings('ignore')

print("Importing TensorFlow/Keras...")
# Import TensorFlow
import tensorflow as tf
print(f"Successfully imported TensorFlow version: {tf.__version__}")

# Enable Metal GPU support for macOS
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Found GPU devices: {physical_devices}")
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU memory growth enabled")
    except Exception as e:
        print(f"Error configuring GPU: {e}")
else:
    print("No GPU devices found, using CPU")

# Import Keras components through TensorFlow
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
print("Successfully imported tensorflow.keras components")

class HealthModelBuilder:
    def __init__(self, processed_data_dir='processed_data', model_dir='saved_models'):
        self.processed_data_dir = processed_data_dir
        self.model_dir = model_dir
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_col = None
        self.model = None
        self.history = None
        self.best_params = None
        self.metrics = {}
        self.imputer = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'figures'), exist_ok=True)
    
    def load_processed_data(self):
        """Load processed data prepared by the HealthDataProcessor"""
        if not os.path.exists(self.processed_data_dir):
            raise ValueError(f"Processed data directory {self.processed_data_dir} does not exist")
        
        # Load the training and test data
        self.X_train = np.load(os.path.join(self.processed_data_dir, 'X_train.npy'))
        self.X_test = np.load(os.path.join(self.processed_data_dir, 'X_test.npy'))
        self.y_train = np.load(os.path.join(self.processed_data_dir, 'y_train.npy'))
        self.y_test = np.load(os.path.join(self.processed_data_dir, 'y_test.npy'))
        
        # Load feature names and target column - check in processed_data first, then data dir
        feature_names_path = os.path.join(self.processed_data_dir, 'feature_names.csv')
        if not os.path.exists(feature_names_path):
            feature_names_path = os.path.join('data', 'feature_names.csv')
            
        target_col_path = os.path.join(self.processed_data_dir, 'target_col.csv')
        if not os.path.exists(target_col_path):
            target_col_path = os.path.join('data', 'target_col.csv')
            
        # Load feature names and target column
        self.feature_names = pd.read_csv(feature_names_path).iloc[:, 0].tolist()
        self.target_col = pd.read_csv(target_col_path).iloc[0, 0]
        
        print(f"Loaded processed data from {self.processed_data_dir}")
        print(f"X_train shape: {self.X_train.shape}")
        print(f"X_test shape: {self.X_test.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Target column: {self.target_col}")
        
        # Check for missing values
        train_nans = np.isnan(self.X_train).sum()
        test_nans = np.isnan(self.X_test).sum()
        
        if train_nans.sum() > 0 or test_nans.sum() > 0:
            print(f"Found {train_nans.sum()} NaN values in training data and {test_nans.sum()} in test data")
            print("Will apply imputation during model training")
            
            # Create imputer
            self.imputer = SimpleImputer(strategy='mean')
            
            # Fit and transform training data
            self.X_train = self.imputer.fit_transform(self.X_train)
            
            # Transform test data
            self.X_test = self.imputer.transform(self.X_test)
            
            # Save the imputer for later use
            joblib.dump(self.imputer, os.path.join(self.processed_data_dir, 'imputer.pkl'))
            print("Imputer saved for future use")
            
        # Check again for missing values
        if np.isnan(self.X_train).sum() > 0 or np.isnan(self.X_test).sum() > 0:
            raise ValueError("Still found NaN values after imputation. Please check your data preprocessing.")
            
        return self
    
    def build_model(self):
        """Build a neural network model using TensorFlow and Keras"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("No data to build model. Load processed data first.")
        
        print("Building neural network model...")
        # Neural network model
        self.model = Sequential()
        self.model.add(Dense(128, activation='relu', input_dim=self.X_train.shape[1]))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))
        
        self.model.add(Dense(64, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        
        self.model.add(Dense(32, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        
        self.model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(learning_rate=0.001)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # Print model summary
        self.model.summary()
        
        # Set the batch size and epochs
        self.batch_size = 32
        self.epochs = 150
        
        return self
    
    def train_model(self, validation_split=0.1, patience=25):
        """Train the model"""
        if self.model is None:
            raise ValueError("No model to train. Build model first.")
        
        print("Training neural network...")
        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint_path = os.path.join(self.model_dir, 'best_model.h5')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train the model
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=validation_split,
            callbacks=[early_stopping, checkpoint, reduce_lr],
            verbose=1
        )
        
        # Load the best model from checkpoint
        if os.path.exists(checkpoint_path):
            self.model = load_model(checkpoint_path)
        
        return self
    
    def evaluate_model(self):
        """Evaluate the model on test data"""
        if self.model is None:
            raise ValueError("No model to evaluate. Train model first.")
        
        # Predict on test data
        y_pred = self.model.predict(self.X_test).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        # Store metrics
        self.metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2)
        }
        
        # Print metrics
        print(f"Model Evaluation Metrics:")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        
        # Plot predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True)
        
        # Save the figure
        figures_dir = os.path.join(self.model_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(os.path.join(figures_dir, 'predictions.png'))
        plt.close()
        
        return self
    
    def plot_training_history(self):
        """Plot training history for neural network models"""
        if self.history is None:
            print("No training history available for plotting")
            return self
        
        # Plot training & validation loss values
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        # Plot training & validation metrics
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'])
        plt.plot(self.history.history['val_mae'])
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        # Save the figure
        figures_dir = os.path.join(self.model_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(os.path.join(figures_dir, 'training_history.png'))
        plt.tight_layout()
        plt.close()
        
        return self
    
    def save_model(self):
        """Save the trained model and associated data"""
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        # Save the feature info
        feature_info = {
            'feature_names': self.feature_names,
            'target_col': self.target_col
        }
        with open(os.path.join(self.model_dir, 'feature_info.json'), 'w') as f:
            json.dump(feature_info, f)
        
        # Save metrics
        with open(os.path.join(self.model_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f)
        
        # Save Keras model
        model_path = os.path.join(self.model_dir, 'health_model.h5')
        self.model.save(model_path)
        print(f"Neural network model saved to {model_path}")
        
        print("Model and associated data saved successfully")
        return self

# Usage example
if __name__ == "__main__":
    # Create model builder
    model_builder = HealthModelBuilder()
    
    # Execute model building pipeline
    (model_builder
     .load_processed_data()
     .build_model()
     .train_model()
     .evaluate_model()
     .plot_training_history()
     .save_model()
    )

    # Set ngrok authentication token
    ngrok.set_auth_token("2uX0aCJoUZuNzfrCZVC3lsvOh5V_4iryQ4V9XANNDiGzrokMH") 