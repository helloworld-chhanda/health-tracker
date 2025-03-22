# AI-Based Health Tracker

An AI-driven health tracking application that uses machine learning to monitor, analyze, and predict health metrics from wearable device data.

## Overview

This application processes health metric data (heart rate, electrodermal activity, temperature, etc.) collected from wearable devices and uses machine learning models to analyze trends, identify patterns, and make predictions. The interactive Streamlit dashboard provides visualizations of health data, personalized predictions, and health recommendations based on the analyzed data.

## Features

- **Data Integration**: Loads and merges health metrics from multiple data sources (HR, EDA, TEMP, BVP)
- **Automated Preprocessing**: Cleans, normalizes, and transforms raw health data for analysis
- **Feature Engineering**: Creates relevant derived features from time-series health data
- **Machine Learning Models**: Uses regression models to analyze patterns and predict health metrics
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
├── create_dummy_model.py    # Alternative model creation script
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
│   └── scaler.pkl           # Fitted scaler for normalization
├── saved_models/            # Directory for trained models
│   ├── health_model.pkl     # Trained machine learning model
│   ├── feature_info.json    # Feature information for the model
│   └── metrics.json         # Model evaluation metrics
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

The application supports different model types:

- **Linear Regression Model**: Fast and interpretable baseline model
- **Neural Network Option**: Deep learning approach with configurable layers and nodes (requires TensorFlow)

Key model features:
- Feature importance analysis
- Performance metrics (MAE, MSE, RMSE, R²)
- Model persistence for future predictions

### Interactive Dashboard

The Streamlit dashboard provides:

- **Dashboard Overview**: Summary statistics and key health metrics visualization
- **Data Explorer**: Interactive tools to analyze health data patterns
- **Predictions**: Interface to input health parameters and receive predictions
- **Model Information**: Details about the model architecture and performance
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

2. Model Training (using create_dummy_model.py for systems with TensorFlow compatibility issues):
```bash
python create_dummy_model.py
```

3. Alternative Model Training (if TensorFlow is properly configured):
```bash
python model.py
```

4. Streamlit App:
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
- tensorflow==2.13.0 (optional, for neural network models)
- keras==2.13.1 (optional, for neural network models)
- streamlit==1.25.0
- plotly==5.15.0
- pyngrok==6.0.0 (optional, for ngrok tunneling)
- joblib==1.3.1
- scikeras==0.11.0 (optional, for neural network models)

## Troubleshooting

- **TensorFlow Issues**: If encountering TensorFlow compatibility problems, use the `create_dummy_model.py` script instead, which creates a scikit-learn based model without TensorFlow dependencies.

- **Model Loading Errors**: Ensure that the model file exists in the `saved_models` directory. If not, run the model training process first.

- **Data Processing Errors**: Check that all required CSV files are present in the project directory.

- **UI Rendering Issues**: If experiencing UI display problems, try adjusting browser zoom or using a different browser.

## Future Enhancements

- Real-time data integration with wearable devices
- Anomaly detection for early warning of health issues
- Mobile application support
- Cloud deployment options
- Personalized health recommendations based on ML insights
- Support for additional health metrics

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Health dataset providers
- Streamlit for the interactive dashboard framework
- The open-source ML community for tools and libraries

## Contact

For questions or support, please contact: [your-email@example.com] # health-tracker
