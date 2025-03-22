# AI-Based Health Tracker

An AI-driven health tracking application that uses machine learning to monitor, analyze, and predict health metrics from wearable device data.

## Overview

This application processes health metric data (heart rate, electrodermal activity, temperature, etc.) collected from wearable devices and uses machine learning models to analyze trends, identify patterns, and make predictions. The interactive Streamlit dashboard provides visualizations of health data, personalized predictions, and health recommendations based on the analyzed data.

## Features

- **Data Integration**: Loads and merges health metrics from multiple data sources (HR, EDA, TEMP, BVP)
- **Automated Preprocessing**: Cleans, normalizes, and transforms raw health data for analysis
- **Feature Engineering**: Creates relevant derived features from time-series health data
- **Adaptive Model Selection**: Automatically selects the appropriate model type (neural network or scikit-learn) based on system compatibility
- **Missing Value Handling**: Automatically detects and imputes missing values in the dataset
- **Interactive Dashboard**: Visualizes health metrics with dynamic charts and graphs
- **Real-time Predictions**: Makes predictions based on user-input health parameters
- **Health Insights**: Provides summary statistics and trend analysis of health data
- **Modern UI**: Clean, responsive interface with intuitive navigation and clear data presentation

## Project Structure

```
health_tracker/
├── app.py                   # Streamlit dashboard application
├── data_preprocessing.py    # Data loading and preprocessing module
├── model.py                 # Machine learning model training module
├── main.py                  # Main script to run the full pipeline
├── setup.sh                 # Setup script for dependencies
├── requirements.txt         # Python dependencies
├── HR.csv                   # Heart rate data
├── EDA.csv                  # Electrodermal activity data
├── TEMP.csv                 # Temperature data
├── BVP.csv                  # Blood volume pulse data
├── IBI.csv                  # Inter-beat interval data
├── ACC.csv                  # Accelerometer data
├── info.txt                 # Information about the dataset
├── processed_data/          # Directory for processed data files
│   ├── X_train.npy          # Training features
│   ├── X_test.npy           # Testing features
│   ├── y_train.npy          # Training target values
│   ├── y_test.npy           # Testing target values
│   ├── feature_names.csv    # Names of features
│   ├── target_col.csv       # Name of target column
│   ├── scaler.pkl           # Fitted scaler for normalization
│   └── imputer.pkl          # Fitted imputer for handling missing values
├── saved_models/            # Directory for trained models
│   ├── health_model.pkl     # Trained scikit-learn model
│   ├── health_model.h5      # Trained neural network model (if available)
│   ├── feature_info.json    # Feature information for the model
│   ├── metrics.json         # Model evaluation metrics
│   └── figures/             # Model visualizations and performance plots
└── visualizations/          # Directory for data visualizations
```

## Technical Details

### Data Processing Pipeline

1. **Data Loading**: Imports data from CSV files containing different health metrics
2. **Data Merging**: Combines data from multiple sources using timestamp alignment
3. **Data Cleaning**: Handles missing values, removes outliers, and normalizes data
4. **Feature Creation**: Generates time-based features and rolling statistics
5. **Data Preparation**: Splits data into training and testing sets for modeling

### Machine Learning Model

The application uses an adaptive model selection approach:

- **Primary Option**: Neural Network with TensorFlow/Keras (when available)
- **Fallback Option**: Linear Regression Model from scikit-learn (when TensorFlow is not available)

Key model features:
- Automatic missing value imputation
- Feature importance analysis
- Comprehensive performance metrics (MAE, MSE, RMSE, R²)
- Model persistence for future predictions
- Visualization of model performance

### Interactive Dashboard

The Streamlit dashboard provides:

- **Dashboard Overview**: Summary statistics and key health metrics visualization
- **Data Explorer**: Interactive tools to analyze health data patterns
- **Predictions**: Interface to input health parameters and receive predictions with visual representation
- **Model Information**: Details about the model architecture, performance, and feature importance
- **Settings**: Application configuration options

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/health_tracker.git
cd health_tracker
```

2. Set up the environment (Option 1 - using setup script):
```bash
chmod +x setup.sh
./setup.sh
```

Or install dependencies manually (Option 2):
```bash
pip install -r requirements.txt
```

For macOS users who want to use TensorFlow with GPU acceleration (optional):
```bash
pip install tensorflow-macos tensorflow-metal
```

## Usage

### Running the Complete Pipeline

To run the complete pipeline (preprocessing, training, and launching the Streamlit app):

```bash
python main.py
```

### Command Line Options

The `main.py` script supports the following options:

- `--skip-preprocess`: Skip data preprocessing stage
- `--skip-train`: Skip model training stage
- `--skip-app`: Skip running the Streamlit app
- `--use-ngrok`: Use ngrok to expose the Streamlit app publicly (requires pyngrok)

Example:
```bash
# Run only the Streamlit app with ngrok tunnel
python main.py --skip-preprocess --skip-train --use-ngrok
```

### Running Components Separately

To run individual components:

1. Data Preprocessing:
```bash
python data_preprocessing.py
```

2. Model Training:
```bash
python model.py
```

3. Streamlit App:
```bash
streamlit run app.py
```

## Data Description

The application uses health metric data with the following components:

- **HR.csv**: Heart rate measurements in beats per minute (BPM)
- **EDA.csv**: Electrodermal activity data measuring skin conductance (µS)
- **TEMP.csv**: Body temperature measurements (°C)
- **BVP.csv**: Blood volume pulse measurements
- **IBI.csv**: Inter-beat interval data measuring time between heartbeats (ms)
- **ACC.csv**: Accelerometer data measuring physical movement

## Performance Metrics

The model is evaluated using several metrics:

- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values
- **MSE (Mean Squared Error)**: Average squared difference between predictions and actual values
- **RMSE (Root Mean Squared Error)**: Square root of MSE, providing error in the same units as the target
- **R² (Coefficient of Determination)**: Proportion of variance in the dependent variable explained by the model

## Requirements

- Python 3.8+
- numpy==1.24.3
- pandas==2.0.3
- matplotlib==3.7.2
- seaborn==0.12.2
- scikit-learn==1.3.0
- streamlit==1.25.0
- plotly==5.15.0
- pyngrok==6.0.0 (optional, for ngrok tunneling)
- joblib==1.3.1
- tensorflow==2.13.0 (optional, for neural network models)
- tensorflow-macos==2.13.0 (optional, for macOS users)
- tensorflow-metal==1.0.0 (optional, for macOS GPU acceleration)
- keras==2.13.1 (optional, for neural network models)

## Troubleshooting

- **Model Selection**: The application will automatically detect if TensorFlow/Keras is available on your system. If not available, it will fall back to using a scikit-learn model, which has fewer dependencies but still provides excellent performance.

- **Missing Values**: The application now automatically detects and handles missing values in the dataset using imputation. No manual preprocessing is needed.

- **TensorFlow on macOS**: For macOS users, we recommend using the `tensorflow-macos` and `tensorflow-metal` packages for better compatibility with Apple hardware:
  ```bash
  pip install tensorflow-macos tensorflow-metal
  ```

- **Model Loading Errors**: If the dashboard cannot load a model, it will now attempt to load an alternative model type (e.g., if `.h5` neural network model fails, it will try the `.pkl` scikit-learn model).

- **Data Processing Errors**: Check that all required CSV files are present in the project directory.

## Future Enhancements

- Real-time data integration with wearable devices
- Anomaly detection for early warning of health issues
- Mobile application support
- Cloud deployment options
- Personalized health recommendations based on ML insights
- Support for additional health metrics
- Enhanced neural network architectures with LSTM layers for time-series analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Health dataset providers
- Streamlit for the interactive dashboard framework
- The open-source ML community for tools and libraries

## Contact

For questions or support, please contact: [your-email@example.com]
