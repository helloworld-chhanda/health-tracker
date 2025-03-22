import numpy as np
import os
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib

def create_dummy_model():
    # Create directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    
    # Load processed data
    X_train = np.load('processed_data/X_train.npy')
    X_test = np.load('processed_data/X_test.npy')
    y_train = np.load('processed_data/y_train.npy')
    y_test = np.load('processed_data/y_test.npy')
    
    print("Data loaded successfully")
    print(f"X_train shape: {X_train.shape}")
    
    # Handle NaN values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    print("NaN values handled with mean imputation")
    
    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X_train_imputed, y_train)
    
    print("Model trained successfully")
    
    # Save the model
    model_path = os.path.join('saved_models', 'health_model.pkl')
    joblib.dump(model, model_path)
    
    # Save feature information
    feature_names = []
    try:
        import pandas as pd
        feature_names = pd.read_csv('processed_data/feature_names.csv').iloc[:, 0].tolist()
        target_col = pd.read_csv('processed_data/target_col.csv').iloc[0, 0]
    except:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        target_col = "heart_rate"
    
    feature_info = {
        'feature_names': feature_names,
        'target_col': target_col
    }
    
    with open(os.path.join('saved_models', 'feature_info.json'), 'w') as f:
        json.dump(feature_info, f)
    
    # Generate and save metrics
    y_pred = model.predict(X_test_imputed)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2)
    }
    
    with open(os.path.join('saved_models', 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    
    print(f"Model and metadata saved in saved_models directory")
    print(f"Model metrics: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

if __name__ == "__main__":
    create_dummy_model() 