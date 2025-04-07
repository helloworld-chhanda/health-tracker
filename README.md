# AI-Based Health Tracker

An AI-driven health tracking application that uses deep learning to monitor, analyze, and predict health metrics from wearable device data.

## Overview

This application processes health metric data (heart rate, electrodermal activity, temperature, etc.) collected from wearable devices and uses neural networks to analyze trends, identify patterns, and make predictions. The interactive Streamlit dashboard provides visualizations of health data, personalized predictions, and health recommendations based on the analyzed data.

## Features

- **Neural Network Model**: Uses TensorFlow/Keras for advanced health metric predictions
- **Interactive Dashboard**: Real-time visualization of health metrics with dynamic charts
- **Personalized Health Analysis**: Provides health status assessment and recommendations
- **Stress Level Analysis**: Monitors and analyzes stress levels using EDA data
- **Activity Tracking**: Tracks and categorizes different physical activities
- **BMI Calculator**: Calculates and tracks Body Mass Index with health recommendations
- **Remote Access**: Supports ngrok tunneling for remote access to the dashboard
- **GPU Acceleration**: Supports Metal GPU acceleration for macOS users

## Project Structure

```
health_tracker/
├── app.py                   # Streamlit dashboard application
├── model.py                 # Neural network model training module
├── requirements.txt         # Python dependencies
├── data/                    # Directory for raw data files
│   ├── HR.csv               # Heart rate data
│   ├── EDA.csv              # Electrodermal activity data
│   └── TEMP.csv             # Temperature data
├── processed_data/          # Directory for processed data files
│   ├── X_train.npy          # Training features
│   ├── X_test.npy           # Testing features
│   ├── y_train.npy          # Training target values
│   ├── y_test.npy           # Testing target values
│   ├── feature_names.csv    # Names of features
│   ├── target_col.csv       # Name of target column
│   ├── scaler.pkl           # Fitted scaler for normalization
│   └── imputer.pkl          # Fitted imputer for handling missing values
└── saved_models/            # Directory for trained models
    ├── health_model.h5      # Trained neural network model
    ├── best_model.h5        # Best model checkpoint during training
    ├── feature_info.json    # Feature information for the model
    ├── metrics.json         # Model evaluation metrics
    └── figures/             # Model visualizations and performance plots
        ├── training_history.png  # Training and validation metrics
        └── predictions.png       # Actual vs predicted values
```

## Technical Details

### Neural Network Architecture

The model uses a deep neural network with:
- 3 Dense layers (128, 64, 32 neurons)
- Batch Normalization for each layer
- Dropout layers for regularization (0.3, 0.2, 0.2)
- Adam optimizer with learning rate reduction
- Early stopping to prevent overfitting
- Mean Squared Error loss function

### Interactive Dashboard

The Streamlit dashboard provides:

- **Dashboard Page**: Overview of key health metrics with interactive visualizations
- **Predictions Page**: Make predictions using the trained neural network model
- **Upload Data Page**: Interface for uploading and processing new health data
- **Model Info Page**: View model architecture, performance metrics, and feature importance
- **About Page**: Information about the application and its features

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/health_tracker.git
cd health_tracker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

For macOS users with GPU support:
```bash
pip install tensorflow-macos tensorflow-metal
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will automatically create an ngrok tunnel if possible.

## Requirements

- Python 3.8+
- numpy==1.24.3
- pandas==2.0.3
- matplotlib==3.7.2
- seaborn==0.12.2
- scikit-learn==1.3.0
- streamlit==1.25.0
- plotly==5.15.0
- pyngrok==6.0.0
- joblib==1.3.1
- tensorflow==2.13.0
- tensorflow-macos==2.13.0 (for macOS users)
- tensorflow-metal==1.0.0 (for macOS GPU acceleration)

## Troubleshooting

- **GPU Support**: The app automatically detects and configures GPU support for macOS users.

- **Ngrok Tunneling**: The app handles ngrok authentication and checks for existing tunnels before creating new ones. If you encounter the "limited to 1 simultaneous ngrok agent sessions" error, the app will continue to run locally.

- **Missing Values**: The application automatically detects and handles missing values in the dataset using imputation.

- **TensorFlow on macOS**: For macOS users, we recommend using the `tensorflow-macos` and `tensorflow-metal` packages for better compatibility with Apple hardware:
  ```bash
  pip install tensorflow-macos tensorflow-metal
  ```

- **Data Processing Errors**: Check that all required CSV files are present in the data directory.

## Future Enhancements

- Real-time data integration with wearable devices
- Anomaly detection for early warning of health issues
- Mobile application support
- Cloud deployment options
- Personalized health recommendations based on ML insights
- Support for additional health metrics
- Enhanced neural network architectures with LSTM layers for time-series analysis

