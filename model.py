import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scikeras.wrappers import KerasRegressor
import json
import time

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
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def load_processed_data(self):
        """Load processed data prepared by the HealthDataProcessor"""
        if not os.path.exists(self.processed_data_dir):
            raise ValueError(f"Processed data directory {self.processed_data_dir} does not exist")
        
        # Load the training and test data
        self.X_train = np.load(os.path.join(self.processed_data_dir, 'X_train.npy'))
        self.X_test = np.load(os.path.join(self.processed_data_dir, 'X_test.npy'))
        self.y_train = np.load(os.path.join(self.processed_data_dir, 'y_train.npy'))
        self.y_test = np.load(os.path.join(self.processed_data_dir, 'y_test.npy'))
        
        # Load feature names and target column
        self.feature_names = pd.read_csv(os.path.join(self.processed_data_dir, 'feature_names.csv')).iloc[:, 0].tolist()
        self.target_col = pd.read_csv(os.path.join(self.processed_data_dir, 'target_col.csv')).iloc[0, 0]
        
        print(f"Loaded processed data from {self.processed_data_dir}")
        print(f"X_train shape: {self.X_train.shape}")
        print(f"X_test shape: {self.X_test.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Target column: {self.target_col}")
        
        return self
    
    def create_base_model(self, input_dim):
        """Create a base model for hyperparameter tuning"""
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=input_dim))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def create_model_function(self, neurons_layer1=64, neurons_layer2=32, dropout_rate=0.2, 
                             learning_rate=0.001, activation='relu'):
        """Create a model with specified hyperparameters"""
        def create_model():
            model = Sequential()
            model.add(Dense(neurons_layer1, activation=activation, input_dim=self.X_train.shape[1]))
            model.add(Dropout(dropout_rate))
            model.add(Dense(neurons_layer2, activation=activation))
            model.add(Dropout(dropout_rate))
            model.add(Dense(1))
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            return model
        return create_model
    
    def tune_hyperparameters(self, cv=3, n_iter=10, random_state=42):
        """Tune hyperparameters using RandomizedSearchCV"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("No data to tune hyperparameters. Load processed data first.")
        
        # Define hyperparameter space
        param_grid = {
            'model__neurons_layer1': [32, 64, 128],
            'model__neurons_layer2': [16, 32, 64],
            'model__dropout_rate': [0.1, 0.2, 0.3],
            'model__learning_rate': [0.0001, 0.001, 0.01],
            'model__activation': ['relu', 'elu', 'selu'],
            'batch_size': [16, 32, 64],
            'epochs': [50, 100]
        }
        
        # Create model wrapper for scikit-learn
        model_function = self.create_model_function()
        model = KerasRegressor(model=model_function, verbose=0)
        
        # Use RandomizedSearchCV for faster tuning
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            verbose=1,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Fit the random search
        start_time = time.time()
        random_search.fit(self.X_train, self.y_train)
        end_time = time.time()
        
        # Store best parameters
        self.best_params = random_search.best_params_
        
        # Print results
        print(f"Hyperparameter tuning completed in {end_time - start_time:.2f} seconds")
        print(f"Best parameters: {self.best_params}")
        print(f"Best score: {random_search.best_score_:.4f}")
        
        # Save best parameters for later use
        with open(os.path.join(self.model_dir, 'best_params.json'), 'w') as f:
            json.dump(self.best_params, f)
        
        return self
    
    def build_model(self):
        """Build model using best hyperparameters or default values"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("No data to build model. Load processed data first.")
        
        if self.best_params is None:
            print("No tuned hyperparameters found. Using default values.")
            # Use default hyperparameters
            neurons_layer1 = 64
            neurons_layer2 = 32
            dropout_rate = 0.2
            learning_rate = 0.001
            activation = 'relu'
            batch_size = 32
            epochs = 100
        else:
            # Use best hyperparameters from tuning
            neurons_layer1 = self.best_params.get('model__neurons_layer1', 64)
            neurons_layer2 = self.best_params.get('model__neurons_layer2', 32)
            dropout_rate = self.best_params.get('model__dropout_rate', 0.2)
            learning_rate = self.best_params.get('model__learning_rate', 0.001)
            activation = self.best_params.get('model__activation', 'relu')
            batch_size = self.best_params.get('batch_size', 32)
            epochs = self.best_params.get('epochs', 100)
        
        # Create model
        self.model = Sequential()
        self.model.add(Dense(neurons_layer1, activation=activation, input_dim=self.X_train.shape[1]))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout_rate))
        
        self.model.add(Dense(neurons_layer2, activation=activation))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout_rate))
        
        self.model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # Print model summary
        self.model.summary()
        
        # Set the batch size and epochs
        self.batch_size = batch_size
        self.epochs = epochs
        
        return self
    
    def train_model(self, validation_split=0.1, patience=20):
        """Train the model with early stopping and learning rate reduction"""
        if self.model is None:
            raise ValueError("No model to train. Build model first.")
        
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
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Save metrics
        with open(os.path.join(self.model_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f)
        
        return self
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            raise ValueError("No training history to plot. Train model first.")
        
        # Create figure directory if it doesn't exist
        figure_dir = os.path.join(self.model_dir, 'figures')
        os.makedirs(figure_dir, exist_ok=True)
        
        # Plot training & validation loss values
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'])
        plt.plot(self.history.history['val_mae'])
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, 'training_history.png'))
        plt.close()
        
        # Plot predicted vs actual values
        y_pred = self.model.predict(self.X_test).flatten()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
        plt.xlabel(f'Actual {self.target_col}')
        plt.ylabel(f'Predicted {self.target_col}')
        plt.title('Actual vs Predicted Values')
        plt.savefig(os.path.join(figure_dir, 'actual_vs_predicted.png'))
        plt.close()
        
        return self
    
    def save_model(self):
        """Save the model and its metadata"""
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        # Save the model
        model_path = os.path.join(self.model_dir, 'health_model.h5')
        self.model.save(model_path)
        
        # Save feature information
        feature_info = {
            'feature_names': self.feature_names,
            'target_col': self.target_col
        }
        
        with open(os.path.join(self.model_dir, 'feature_info.json'), 'w') as f:
            json.dump(feature_info, f)
        
        print(f"Model and metadata saved in {self.model_dir}")
        
        return self

# Usage example
if __name__ == "__main__":
    # Create model builder
    model_builder = HealthModelBuilder()
    
    # Execute model building pipeline
    (model_builder
     .load_processed_data()
     .tune_hyperparameters(cv=3, n_iter=5)  # Reduced iterations for faster execution
     .build_model()
     .train_model()
     .evaluate_model()
     .plot_training_history()
     .save_model()
    ) 