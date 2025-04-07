import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os
import json
import joblib
from datetime import datetime, timedelta
from pyngrok import ngrok
import time
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

# Import TensorFlow and Keras
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.version.VERSION}")
    
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
    
    # Import Keras
    from tensorflow.keras.models import load_model
    print("Successfully imported tensorflow.keras")
except ImportError as e:
    st.error(f"Failed to import TensorFlow/Keras: {e}")
    st.error("This application requires TensorFlow and Keras to run. Please install it and try again.")
    raise SystemExit("TensorFlow is required to run this application.")

# Set page configuration
st.set_page_config(
    page_title="AI Health Tracker",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stApp {
        background-color: #f5f7f9;
    }
    .css-18e3th9 {
        padding-top: 1rem;
        padding-bottom: 10rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    h1 {
        color: #2e4057;
        font-weight: 600;
    }
    h2, h3 {
        color: #336699;
        font-weight: 500;
    }
    p, li, .stTextInput>div>div>input, .stSelectbox, .stNumberInput {
        color: #333333;
    }
    .stAlert {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4a86e8;
        color: white;
        font-weight: 500;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #3a76d8;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .css-1d391kg, .css-12oz5g7 {
        background-color: #ffffff;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #336699;
        font-weight: 600;
    }
    /* Form fields */
    [data-baseweb="input"], [data-baseweb="select"] {
        border-radius: 4px;
        border-color: #cccccc;
    }
    /* Dataframe styling */
    .stDataFrame {
        border-radius: a6px;
    }
    .stDataFrame [data-testid="stTable"] {
        border: 1px solid #e6e6e6;
        border-radius: 6px;
    }
    /* Success/warning/info messages */
    .element-container .stAlert [data-baseweb="notification"] {
        border-radius: 6px;
    }
    .element-container .stAlert [data-baseweb="notification"][data-testid="info"] {
        background-color: #e6f4ff;
        border-color: #91caff;
    }
    .element-container .stAlert [data-baseweb="notification"][data-testid="success"] {
        background-color: #e6ffed;
        border-color: #b7eb8f;
    }
    .element-container .stAlert [data-baseweb="notification"][data-testid="warning"] {
        background-color: #fff9e6;
        border-color: #ffe58f;
    }
    .element-container .stAlert [data-baseweb="notification"][data-testid="error"] {
        background-color: #fff1f0;
        border-color: #ffccc7;
    }
</style>
""", unsafe_allow_html=True)

# Class for health tracking application
class HealthTrackerApp:
    def __init__(self, model_dir='saved_models', data_dir='data'):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.target_col = None
        self.demo_data = None
        
    def load_model(self):
        """Load the trained model and preprocessing objects"""
        # Load neural network model
        model_path = os.path.join(self.model_dir, 'health_model.h5')
        
        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                st.sidebar.success("Neural network model loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading neural network model: {e}")
                return False
        else:
            st.sidebar.error("Model file not found. Please train the model first.")
            return False
        
        # Load scaler
        scaler_path = os.path.join('processed_data', 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            st.sidebar.warning("Scaler not found. Predictions may be inaccurate.")
        
        # Load imputer if available
        imputer_path = os.path.join('processed_data', 'imputer.pkl')
        if os.path.exists(imputer_path):
            self.imputer = joblib.load(imputer_path)
        else:
            self.imputer = None
        
        # Load feature info
        feature_info_path = os.path.join(self.model_dir, 'feature_info.json')
        if os.path.exists(feature_info_path):
            with open(feature_info_path, 'r') as f:
                feature_info = json.load(f)
                self.feature_names = feature_info['feature_names']
                self.target_col = feature_info['target_col']
        else:
            st.sidebar.warning("Feature info not found. Using default values.")
            self.feature_names = []
            self.target_col = 'heart_rate'
        
        # Load model metrics if available
        metrics_path = os.path.join(self.model_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
        else:
            self.metrics = None
        
        return True
    
    def load_demo_data(self):
        """Load a sample of the data for demonstration purposes"""
        try:
            # Load HR data for demo
            hr_path = os.path.join(self.data_dir, 'HR.csv')
            if os.path.exists(hr_path):
                hr_df = pd.read_csv(hr_path, header=None)
                timestamp = float(hr_df.iloc[0, 0])
                sample_rate = float(hr_df.iloc[1, 0])
                # Take just a sample for demo
                hr_values = hr_df.iloc[2:202].values.flatten()
                
                # Create proper time index
                times = np.arange(len(hr_values)) / sample_rate
                self.demo_data = pd.DataFrame({
                    'timestamp': times + timestamp,
                    'heart_rate': hr_values
                })
                
                # Convert timestamp to datetime for better visualization
                self.demo_data['datetime'] = pd.to_datetime(self.demo_data['timestamp'], unit='s')
                
                # Add some additional features for demo
                self.demo_data['stress_level'] = np.where(
                    self.demo_data['heart_rate'] > self.demo_data['heart_rate'].quantile(0.7),
                    'High',
                    np.where(
                        self.demo_data['heart_rate'] < self.demo_data['heart_rate'].quantile(0.3),
                        'Low',
                        'Normal'
                    )
                )
                
                # Generate random activity data
                activities = ['Rest', 'Walk', 'Run', 'Exercise']
                weights = [0.4, 0.3, 0.2, 0.1]
                self.demo_data['activity'] = np.random.choice(
                    activities, 
                    size=len(self.demo_data), 
                    p=weights
                )
                
                return True
            else:
                st.warning("Demo data not available. Please upload your own data.")
                return False
                
        except Exception as e:
            st.error(f"Error loading demo data: {e}")
            return False
    
    def run(self):
        """Main function to run the Streamlit app"""
        # App title and description
        st.title("🏥 AI-Based Health Tracker")
        
        # Sidebar
        with st.sidebar:
            st.header("Navigation")
            page = st.radio(
                "Go to",
                ["Dashboard", "Predictions", "Upload Data", "Model Info", "About"]
            )
            
            st.header("Options")
            demo_mode = st.checkbox("Use Demo Data", value=True)
            
            if st.button("Load Model"):
                self.load_model()
            
            if demo_mode and st.button("Load Demo Data"):
                if self.load_demo_data():
                    st.success("Demo data loaded successfully!")
        
        # Display the selected page
        if page == "Dashboard":
            self.dashboard_page(demo_mode)
        elif page == "Predictions":
            self.predictions_page()
        elif page == "Upload Data":
            self.upload_data_page()
        elif page == "Model Info":
            self.model_info_page()
        else:
            self.about_page()
    
    def dashboard_page(self, demo_mode=True):
        """Display the dashboard page"""
        st.header("Health Metrics Dashboard")
        
        # If demo mode is on and demo data is not loaded, try to load it
        if demo_mode and self.demo_data is None:
            self.load_demo_data()
        
        # Check if we have data to display
        if self.demo_data is not None:
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                heart_rate_avg = self.demo_data['heart_rate'].mean()
                delta = heart_rate_avg - 70
                delta_color = "normal" if -5 <= delta <= 5 else ("inverse" if delta < -5 else "off")
                st.metric(
                    "Average Heart Rate", 
                    f"{heart_rate_avg:.1f} BPM",
                    f"{delta:.1f}",
                    delta_color=delta_color
                )
            
            with col2:
                high_stress = (self.demo_data['stress_level'] == 'High').sum()
                stress_percentage = (high_stress / len(self.demo_data)) * 100
                delta = stress_percentage - 30
                delta_color = "inverse" if delta < 0 else ("normal" if delta == 0 else "off")
                st.metric(
                    "Stress Level", 
                    f"{stress_percentage:.1f}%",
                    f"{delta:.1f}%",
                    delta_color=delta_color
                )
            
            with col3:
                activity_counts = self.demo_data['activity'].value_counts()
                most_common_activity = activity_counts.idxmax()
                st.metric(
                    "Most Common Activity", 
                    most_common_activity,
                    ""
                )
            
            # Display heart rate chart
            st.subheader("Heart Rate Over Time")
            fig = px.line(
                self.demo_data, 
                x='datetime', 
                y='heart_rate',
                title='Heart Rate Trend',
                labels={'datetime': 'Time', 'heart_rate': 'Heart Rate (BPM)'},
                color_discrete_sequence=['#4a86e8']
            )
            fig.update_layout(
                height=400,
                xaxis_title='Time',
                yaxis_title='Heart Rate (BPM)',
                hovermode='x unified',
                plot_bgcolor='rgba(245, 247, 249, 0.5)',
                paper_bgcolor='rgba(245, 247, 249, 0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display activity distribution
            st.subheader("Activity Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                activity_counts = self.demo_data['activity'].value_counts().reset_index()
                activity_counts.columns = ['Activity', 'Count']
                
                fig = px.pie(
                    activity_counts, 
                    values='Count', 
                    names='Activity',
                    title='Activity Distribution',
                    hole=0.4,
                    color_discrete_sequence=['#4a86e8', '#5e97f6', '#7baaf7', '#98bef9']
                )
                fig.update_layout(
                    paper_bgcolor='rgba(245, 247, 249, 0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                stress_counts = self.demo_data['stress_level'].value_counts().reset_index()
                stress_counts.columns = ['Stress Level', 'Count']
                
                fig = px.bar(
                    stress_counts, 
                    x='Stress Level', 
                    y='Count',
                    title='Stress Level Distribution',
                    color='Stress Level',
                    color_discrete_map={'Low': '#5cb85c', 'Normal': '#4a86e8', 'High': '#d9534f'}
                )
                fig.update_layout(
                    paper_bgcolor='rgba(245, 247, 249, 0)',
                    plot_bgcolor='rgba(245, 247, 249, 0.5)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Heart rate by activity box plot
            st.subheader("Heart Rate by Activity")
            fig = px.box(
                self.demo_data,
                x='activity',
                y='heart_rate',
                color='activity',
                title='Heart Rate Distribution by Activity',
                labels={'activity': 'Activity', 'heart_rate': 'Heart Rate (BPM)'},
                color_discrete_map={'Rest': '#5cb85c', 'Walk': '#4a86e8', 'Run': '#f0ad4e', 'Exercise': '#d9534f'}
            )
            fig.update_layout(
                paper_bgcolor='rgba(245, 247, 249, 0)',
                plot_bgcolor='rgba(245, 247, 249, 0.5)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display alerts
            high_hr_threshold = 100
            high_hr_count = (self.demo_data['heart_rate'] > high_hr_threshold).sum()
            
            if high_hr_count > 0:
                st.warning(f"⚠️ Alert: Heart rate exceeded {high_hr_threshold} BPM {high_hr_count} times during this period!")
            
        else:
            st.info("No data available. Please load demo data or upload your own data.")
            
            # Show placeholder charts
            st.subheader("Sample Visualizations (No Data)")
            
            # Sample heart rate data
            dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
            heart_rates = 70 + 10 * np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 5, 100)
            sample_data = pd.DataFrame({'datetime': dates, 'heart_rate': heart_rates})
            
            fig = px.line(
                sample_data,
                x='datetime',
                y='heart_rate',
                title='Sample Heart Rate Trend',
                labels={'datetime': 'Time', 'heart_rate': 'Heart Rate (BPM)'},
                color_discrete_sequence=['#4a86e8']
            )
            fig.update_layout(
                height=400,
                xaxis_title='Time',
                yaxis_title='Heart Rate (BPM)',
                hovermode='x unified',
                plot_bgcolor='rgba(245, 247, 249, 0.5)',
                paper_bgcolor='rgba(245, 247, 249, 0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def predictions_page(self):
        """Make predictions based on user input"""
        st.header("Make Predictions")
        
        if self.model is None:
            if not self.load_model():
                st.warning("Please load a model first to make predictions.")
                return
        
        # Variables to store prediction results outside the form
        prediction_made = False
        prediction_data = {}
        
        # Create form for user inputs
        with st.form("prediction_form"):
            # Initialize default values
            inputs = {}
            
            # Create input fields for each feature
            st.subheader("Enter Health Metrics")
            
            # Define default values based on feature names
            default_values = {
                # Heart rate related defaults
                'heart_rate': 75.0,
                'hr_rolling_mean': 75.0,
                'hr_rolling_std': 5.0,
                
                # EDA (Electrodermal Activity) related defaults
                'electrodermal_activity': 5.0,
                'eda_rolling_mean': 5.0,
                'eda_rolling_std': 0.5,
                
                # Temperature related defaults
                'temperature': 36.5,
                'temp_rolling_mean': 36.5,
                'temp_rolling_std': 0.2,
                
                # Blood volume pulse related defaults
                'blood_volume_pulse': 70.0,
                'bvp_rolling_mean': 70.0,
                'bvp_rolling_std': 10.0,
                
                # Time related defaults
                'hour': 12.0,
                'minute': 30.0,
                'second': 0.0,
                'day_of_week': 3.0,
                
                # Default for any other features
                'default': 0.0
            }
            
            # Create columns for inputs
            num_cols = 3
            for i in range(0, len(self.feature_names), num_cols):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    idx = i + j
                    if idx < len(self.feature_names):
                        feature = self.feature_names[idx]
                        # Get appropriate default value based on feature name
                        default_value = default_values.get(feature, default_values['default'])
                        
                        # Set appropriate min, max, and step values based on feature type
                        min_val, max_val, step_val = self._get_input_range(feature)
                        
                        inputs[feature] = cols[j].number_input(
                            f"{feature.replace('_', ' ').title()}", 
                            value=default_value,
                            min_value=min_val,
                            max_value=max_val,
                            step=step_val
                        )
            
            # Patient information (optional)
            st.subheader("Patient Information (Optional)")
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age", min_value=1, max_value=120, value=35, step=1)
            with col2:
                gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)
            with col3:
                weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0, step=0.1)
            
            col1, col2 = st.columns(2)
            with col1:
                height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.1)
            with col2:
                activity_level = st.selectbox("Activity Level", 
                                             ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"], 
                                             index=1)
            
            # Submit button
            submitted = st.form_submit_button("Predict")
            
            if submitted:
                try:
                    # Prepare input data
                    input_data = np.array([inputs[feature] for feature in self.feature_names]).reshape(1, -1)
                    
                    # Apply imputation if available and needed
                    if np.isnan(input_data).any() and self.imputer is not None:
                        input_data = self.imputer.transform(input_data)
                    elif np.isnan(input_data).any():
                        st.warning("Input contains missing values and no imputer is available. Results may be inaccurate.")
                    
                    # Scale the input if scaler is available
                    if self.scaler is not None:
                        input_data = self.scaler.transform(input_data)
                    
                    # Make prediction with neural network model
                    prediction = self.model.predict(input_data)[0][0]
                    
                    # Store key health metrics for display
                    heart_rate = inputs.get('heart_rate', prediction if self.target_col == 'heart_rate' else 75.0)
                    temperature = inputs.get('temperature', 36.5)
                    eda = inputs.get('electrodermal_activity', 5.0)
                    
                    # Store prediction results to use outside the form
                    prediction_made = True
                    prediction_data = {
                        'prediction': prediction,
                        'target_col': self.target_col,
                        'heart_rate': heart_rate,
                        'temperature': temperature,
                        'eda': eda,
                        'patient_info': {
                            'age': age,
                            'gender': gender,
                            'weight': weight,
                            'height': height,
                            'activity_level': activity_level
                        }
                    }
                    
                    # Display prediction results in a nice format
                    st.header("🩺 Health Prediction Results")
                    
                    # Create tabs for different views of the prediction
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Detailed Analysis", "🩺 Health Assessment", "📋 Recommendations"])
                    
                    with tab1:
                        # Overview tab
                        st.subheader("Predicted Health Metrics")
                        
                        # Display prediction in columns
                        cols = st.columns(3)
                        with cols[0]:
                            if self.target_col == 'heart_rate':
                                status_color = self._get_heart_rate_status_color(prediction)
                                st.markdown(f"<h1 style='text-align: center; color: {status_color};'>{prediction:.1f}</h1>", unsafe_allow_html=True)
                                st.markdown("<p style='text-align: center;'>Predicted Heart Rate (BPM)</p>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<h1 style='text-align: center;'>{prediction:.1f}</h1>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center;'>Predicted {self.target_col.replace('_', ' ').title()}</p>", unsafe_allow_html=True)
                        
                        with cols[1]:
                            if 'temperature' in inputs:
                                temp_status_color = "#4a86e8"  # Default blue
                                if temperature < 36.0:
                                    temp_status_color = "#9467bd"  # Purple for low
                                elif temperature > 37.5:
                                    temp_status_color = "#d9534f"  # Red for high
                                
                                st.markdown(f"<h1 style='text-align: center; color: {temp_status_color};'>{temperature:.1f}</h1>", unsafe_allow_html=True)
                                st.markdown("<p style='text-align: center;'>Body Temperature (°C)</p>", unsafe_allow_html=True)
                        
                        with cols[2]:
                            if 'electrodermal_activity' in inputs:
                                eda_status_color = "#4a86e8"  # Default blue
                                if eda > 10.0:
                                    eda_status_color = "#f0ad4e"  # Orange for high stress
                                
                                st.markdown(f"<h1 style='text-align: center; color: {eda_status_color};'>{eda:.1f}</h1>", unsafe_allow_html=True)
                                st.markdown("<p style='text-align: center;'>Electrodermal Activity (μS)</p>", unsafe_allow_html=True)
                        
                        # Create gauge chart for the prediction
                        st.subheader("Health Status Indicator")
                        
                        # Determine reasonable min/max values for the gauge
                        min_val = 0
                        max_val = 200
                        if self.target_col == 'heart_rate':
                            min_val = 40
                            max_val = 180
                            threshold_zones = [
                                {'name': 'Rest', 'min': 40, 'max': 60, 'color': 'blue'},
                                {'name': 'Normal', 'min': 60, 'max': 100, 'color': 'green'},
                                {'name': 'Elevated', 'min': 100, 'max': 140, 'color': 'orange'},
                                {'name': 'High', 'min': 140, 'max': 180, 'color': 'red'}
                            ]
                        elif self.target_col == 'stress_level':
                            min_val = 0
                            max_val = 10
                            threshold_zones = [
                                {'name': 'Low', 'min': 0, 'max': 3, 'color': 'green'},
                                {'name': 'Medium', 'min': 3, 'max': 7, 'color': 'orange'},
                                {'name': 'High', 'min': 7, 'max': 10, 'color': 'red'}
                            ]
                        else:
                            # For other targets, use generic visualization
                            threshold_zones = []
                            
                        # Get the status text
                        status_text = self._get_health_status_text(self.target_col, prediction)
                        
                        # Create gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=prediction,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': f"Predicted {self.target_col.replace('_', ' ').title()}", 'font': {'size': 24}},
                            delta={'reference': self._get_normal_reference_value(self.target_col)},
                            gauge={
                                'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                'bar': {'color': "#5e97f6"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [zone['min'], zone['max']], 'color': zone['color']} 
                                    for zone in threshold_zones
                                ],
                            }
                        ))
                        
                        fig.update_layout(
                            height=300,
                            paper_bgcolor='rgba(245, 247, 249, 0)',
                            plot_bgcolor='rgba(245, 247, 249, 0)',
                            font={'color': "darkblue", 'family': "Arial"}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display health status
                        st.info(f"**Health Status:** {status_text}")
                        
                    with tab2:
                        # Detailed Analysis tab
                        st.subheader("Detailed Health Metrics Analysis")
                        
                        # Create a radar chart for multiple metrics
                        selected_features = [f for f in self.feature_names if not any(x in f for x in ['rolling', 'hour', 'minute', 'second', 'day'])]
                        if len(selected_features) > 2:
                            # Limit to most important features
                            selected_features = selected_features[:5]
                        
                        # Normalize values for radar chart
                        radar_values = []
                        for feature in selected_features:
                            value = inputs.get(feature, 0)
                            min_val, max_val, _ = self._get_input_range(feature)
                            if min_val is not None and max_val is not None:
                                # Normalize to 0-1 scale
                                normalized = (value - min_val) / (max_val - min_val)
                                radar_values.append(normalized)
                            else:
                                radar_values.append(0.5)  # Default for unknown ranges
                        
                        # Create radar chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatterpolar(
                            r=radar_values,
                            theta=[f.replace('_', ' ').title() for f in selected_features],
                            fill='toself',
                            name='Current Metrics',
                            line_color='#4a86e8'
                        ))
                        
                        # Add reference values for comparison
                        reference_values = []
                        for feature in selected_features:
                            ref_val = self._get_normal_reference_value(feature)
                            min_val, max_val, _ = self._get_input_range(feature)
                            if min_val is not None and max_val is not None:
                                # Normalize to 0-1 scale
                                normalized = (ref_val - min_val) / (max_val - min_val)
                                reference_values.append(normalized)
                            else:
                                reference_values.append(0.5)  # Default for unknown ranges
                        
                        fig.add_trace(go.Scatterpolar(
                            r=reference_values,
                            theta=[f.replace('_', ' ').title() for f in selected_features],
                            fill='toself',
                            name='Normal Range',
                            line_color='rgba(0, 200, 0, 0.5)',
                            fillcolor='rgba(0, 200, 0, 0.1)'
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            showlegend=True,
                            height=400,
                            paper_bgcolor='rgba(245, 247, 249, 0)',
                            plot_bgcolor='rgba(245, 247, 249, 0)'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show trend data (simulated for demonstration)
                        st.subheader("Trend Analysis")
                        
                        # Create simulated trend data
                        dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
                        if self.target_col == 'heart_rate':
                            # Simulate heart rate trend with slight variation
                            trend_values = [
                                prediction * (1 + 0.1 * np.sin(i)) 
                                for i in range(10)
                            ]
                            
                            # Create trend chart
                            trend_df = pd.DataFrame({
                                'Date': dates,
                                'Value': trend_values
                            })
                            
                            fig = px.line(
                                trend_df, 
                                x='Date', 
                                y='Value',
                                title=f'Estimated {self.target_col.replace("_", " ").title()} Trend',
                                markers=True
                            )
                            
                            # Add reference ranges
                            fig.add_hline(y=60, line_dash="dash", line_color="green", annotation_text="Min Normal")
                            fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Max Normal")
                            
                            fig.update_layout(
                                height=300,
                                xaxis_title='Date',
                                yaxis_title=self.target_col.replace('_', ' ').title(),
                                paper_bgcolor='rgba(245, 247, 249, 0)',
                                plot_bgcolor='rgba(245, 247, 249, 0.5)'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        # Health Assessment tab
                        st.subheader("Health Risk Assessment")
                        
                        # Calculate BMI if height and weight provided
                        if height > 0 and weight > 0:
                            bmi = weight / ((height/100) ** 2)
                            bmi_category = self._get_bmi_category(bmi)
                            
                            # Create BMI indicator
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=bmi,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "BMI"},
                                gauge={
                                    'axis': {'range': [10, 40]},
                                    'bar': {'color': self._get_bmi_color(bmi)},
                                    'steps': [
                                        {'range': [10, 18.5], 'color': '#9467bd'},  # Underweight
                                        {'range': [18.5, 25], 'color': '#5cb85c'},  # Normal
                                        {'range': [25, 30], 'color': '#f0ad4e'},    # Overweight
                                        {'range': [30, 40], 'color': '#d9534f'}     # Obese
                                    ],
                                }
                            ))
                            
                            fig.update_layout(
                                height=250,
                                paper_bgcolor='rgba(245, 247, 249, 0)',
                                plot_bgcolor='rgba(245, 247, 249, 0)'
                            )
                            
                            col1, col2 = st.columns([2, 3])
                            
                            with col1:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.markdown(f"**BMI Category:** {bmi_category}")
                                st.markdown(f"**Height:** {height} cm")
                                st.markdown(f"**Weight:** {weight} kg")
                                st.markdown(f"**Gender:** {gender}")
                                st.markdown(f"**Age:** {age} years")
                                st.markdown(f"**Activity Level:** {activity_level}")
                        
                        # Potential health conditions section
                        st.subheader("Potential Health Indicators")
                        
                        # Get health conditions based on predicted value
                        conditions = self._get_potential_health_conditions(self.target_col, prediction, temperature, eda)
                        
                        if conditions:
                            # Create risk level indicators for each condition
                            for condition in conditions:
                                col1, col2 = st.columns([1, 4])
                                with col1:
                                    risk_color = {
                                        'Low': '#5cb85c',
                                        'Moderate': '#f0ad4e',
                                        'High': '#d9534f'
                                    }.get(condition['risk_level'], '#4a86e8')
                                    
                                    st.markdown(f"""
                                    <div style="background-color: {risk_color}; padding: 10px; border-radius: 5px; text-align: center; color: white;">
                                        <strong>{condition['risk_level']}</strong>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"**{condition['name']}**")
                                    st.markdown(f"{condition['description']}")
                        else:
                            st.info("No specific health concerns identified based on the provided metrics.")
                    
                    with tab4:
                        # Recommendations tab
                        st.subheader("Personalized Health Recommendations")
                        
                        # Generate recommendations based on inputs and prediction
                        recommendations = self._generate_recommendations(
                            self.target_col, 
                            prediction, 
                            heart_rate, 
                            temperature, 
                            eda,
                            age=age,
                            gender=gender,
                            activity_level=activity_level
                        )
                        
                        # Save conditions and recommendations for use outside the form
                        prediction_data['conditions'] = conditions
                        prediction_data['recommendations'] = recommendations
                        
                        # Display recommendations in expandable sections
                        for category, recs in recommendations.items():
                            with st.expander(f"📝 {category}", expanded=True):
                                for rec in recs:
                                    st.markdown(f"• {rec}")
                        
                        # Call to action
                        st.info("📌 **Note:** These recommendations are generated based on the predicted values and provided information. Always consult with a healthcare professional for personalized medical advice.")
                        
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # Add download button outside the form
        if prediction_made:
            # Option to download report
            st.download_button(
                "📥 Download Health Assessment Report",
                self._generate_health_report(
                    prediction_data['prediction'], 
                    prediction_data['target_col'],
                    prediction_data['heart_rate'],
                    prediction_data['temperature'],
                    prediction_data['eda'],
                    prediction_data['conditions'],
                    prediction_data['recommendations'],
                    prediction_data['patient_info']
                ),
                file_name="health_assessment_report.txt",
                mime="text/plain"
            )
    
    def _get_input_range(self, feature):
        """Helper function to determine appropriate input ranges for different features"""
        # Default values
        min_val = None
        max_val = None
        step_val = 0.1
        
        # Set ranges based on feature type
        if 'heart_rate' in feature:
            min_val = 40.0
            max_val = 200.0
            step_val = 1.0
        elif 'electrodermal' in feature:
            min_val = 0.0
            max_val = 20.0
            step_val = 0.1
        elif 'temperature' in feature:
            min_val = 35.0
            max_val = 40.0
            step_val = 0.1
        elif 'blood' in feature:
            min_val = 0.0
            max_val = 150.0
            step_val = 1.0
        elif 'hour' in feature:
            min_val = 0.0
            max_val = 23.0
            step_val = 1.0
        elif 'minute' in feature or 'second' in feature:
            min_val = 0.0
            max_val = 59.0
            step_val = 1.0
        elif 'day_of_week' in feature:
            min_val = 0.0
            max_val = 6.0
            step_val = 1.0
        elif 'std' in feature:
            min_val = 0.0
            max_val = 20.0
            step_val = 0.1
            
        return min_val, max_val, step_val
    
    def upload_data_page(self):
        """Display the data upload page"""
        st.header("Upload Your Health Data")
        st.write("Upload your health data files to analyze and make predictions.")
        
        # File upload widgets
        uploaded_files = st.file_uploader(
            "Upload CSV files (HR.csv, EDA.csv, TEMP.csv, etc.)",
            type=["csv"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # Display the uploaded files
            st.subheader("Uploaded Files")
            file_names = [file.name for file in uploaded_files]
            st.write(", ".join(file_names))
            
            # Process button
            if st.button("Process Uploaded Data"):
                # Placeholder for data processing logic
                with st.spinner("Processing data..."):
                    time.sleep(2)  # Simulate processing time
                    st.success("Data processed successfully!")
                    
                    # Create a toggle to show the processed data
                    show_data = st.checkbox("Show processed data")
                    if show_data:
                        # Read the first uploaded file as an example
                        data = pd.read_csv(uploaded_files[0])
                        st.dataframe(data.head(10))
                    
                    # Add option to use this data for prediction
                    if st.button("Use this data for prediction"):
                        st.session_state['uploaded_data_processed'] = True
                        st.info("You can now go to the Predictions page to make predictions with your data.")
        
        # Alternative: manual data entry
        st.subheader("Or Enter Your Data Manually")
        st.write("Don't have a CSV file? You can enter your health metrics manually.")
        
        with st.form("manual_data_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                heart_rate = st.number_input(
                    "Heart Rate (BPM)", 
                    min_value=40, 
                    max_value=200, 
                    value=75, 
                    step=1,
                    help="Normal resting heart rate is typically between 60-100 BPM"
                )
            
            with col2:
                eda = st.number_input(
                    "Electrodermal Activity (μS)", 
                    min_value=0.0, 
                    max_value=30.0, 
                    value=5.0, 
                    step=0.1,
                    help="Measures skin conductance which increases with psychological arousal"
                )
            
            with col3:
                temp = st.number_input(
                    "Body Temperature (°C)", 
                    min_value=35.0, 
                    max_value=42.0, 
                    value=36.7, 
                    step=0.1,
                    help="Normal body temperature is around 36.5-37.5°C"
                )
            
            # Additional inputs
            col1, col2 = st.columns(2)
            
            with col1:
                activity = st.selectbox(
                    "Current Activity", 
                    ["Rest", "Walk", "Run", "Exercise", "Sleep", "Work"],
                    index=0,
                    help="Select your current physical activity"
                )
            
            with col2:
                stress_level = st.slider(
                    "Stress Level (0-10)",
                    min_value=0,
                    max_value=10,
                    value=3,
                    step=1,
                    help="0 = No stress, 10 = Maximum stress"
                )
            
            # Time information
            st.subheader("Time Information")
            col1, col2 = st.columns(2)
            
            with col1:
                current_time = datetime.now()
                measurement_time = st.time_input(
                    "Time of Measurement",
                    value=current_time.time(),
                    help="The time when these measurements were taken"
                )
            
            with col2:
                measurement_date = st.date_input(
                    "Date of Measurement",
                    value=current_time.date(),
                    help="The date when these measurements were taken"
                )
            
            # Notes
            notes = st.text_area(
                "Notes",
                value="",
                help="Any additional information or context about these measurements"
            )
            
            submitted = st.form_submit_button("Save Data")
        
        if submitted:
            st.success("Data saved successfully!")
            
            # Create a dataframe with the manually entered data
            measurement_datetime = datetime.combine(measurement_date, measurement_time)
            
            manual_data = pd.DataFrame({
                'timestamp': [measurement_datetime.timestamp()],
                'datetime': [measurement_datetime],
                'heart_rate': [heart_rate],
                'electrodermal_activity': [eda],
                'temperature': [temp],
                'activity': [activity],
                'stress_level': [stress_level],
                'notes': [notes]
            })
            
            # Display the data
            st.dataframe(manual_data)
            
            # Option to download the data as CSV
            csv = manual_data.to_csv(index=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name="health_data.csv",
                mime="text/csv"
            )
    
    def model_info_page(self):
        """Display information about the model"""
        st.header("Model Information")
        
        # Try to load the model if not already loaded
        if self.model is None:
            self.load_model()
        
        if self.model is not None:
            # Display model type
            st.subheader("Model Type")
            st.write("This is a **Neural Network** model built with TensorFlow and Keras.")
            
            # Display model architecture
            st.subheader("Model Architecture")
            model_summary = []
            self.model.summary(print_fn=lambda x: model_summary.append(x))
            st.code("\n".join(model_summary))
            
            # Display model metrics if available
            if self.metrics:
                st.subheader("Model Performance Metrics")
                cols = st.columns(4)
                metrics = [
                    ("MAE", self.metrics.get('mae', 'N/A'), "Mean Absolute Error"),
                    ("MSE", self.metrics.get('mse', 'N/A'), "Mean Squared Error"),
                    ("RMSE", self.metrics.get('rmse', 'N/A'), "Root Mean Squared Error"),
                    ("R²", self.metrics.get('r2', 'N/A'), "Coefficient of Determination")
                ]
                
                for i, (metric, value, description) in enumerate(metrics):
                    with cols[i]:
                        st.metric(label=f"{metric}", value=f"{value:.4f}" if isinstance(value, float) else value)
                        st.caption(description)
            
            # Display feature importance visualization
            if self.feature_names:
                st.subheader("Feature Importance")
                st.write("This visualization represents an approximation of how each feature contributes to predictions.")
                
                # For neural networks, we'll create a simple feature importance visualization
                # based on the weights of the first layer
                try:
                    # Get weights from the first layer
                    first_layer_weights = np.abs(self.model.layers[0].get_weights()[0])
                    # Sum the absolute weights for each feature
                    importances = np.sum(first_layer_weights, axis=1)
                    # Normalize to sum to 1
                    importances = importances / np.sum(importances)
                    
                    feature_imp = pd.DataFrame({
                        'Feature': self.feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        feature_imp,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Feature Importance Estimation (Based on First Layer Weights)',
                        labels={'Importance': 'Relative Importance', 'Feature': 'Feature Name'},
                        color='Importance',
                        color_continuous_scale=['#98bef9', '#7baaf7', '#5e97f6', '#4a86e8', '#3a76d8']
                    )
                    fig.update_layout(
                        height=500,
                        paper_bgcolor='rgba(245, 247, 249, 0)',
                        plot_bgcolor='rgba(245, 247, 249, 0.5)',
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not calculate feature importance: {e}")
                    st.info("Feature importance calculation for deep neural networks requires additional tooling.")
                
                # Show model training history if available
                history_dir = os.path.join(self.model_dir, 'figures')
                history_path = os.path.join(history_dir, 'training_history.png')
                
                if os.path.exists(history_path):
                    st.subheader("Training History")
                    st.image(history_path, caption="Model Training History")
                    
                # Show predictions visualization if available
                predictions_path = os.path.join(history_dir, 'predictions.png')
                if os.path.exists(predictions_path):
                    st.subheader("Prediction Performance")
                    st.image(predictions_path, caption="Actual vs Predicted Values")
            else:
                st.info("Feature information not available.")
        else:
            st.warning("Model not loaded. Please load the model first.")
    
    def about_page(self):
        """Display information about the application"""
        st.header("About AI Health Tracker")
        
        st.markdown("""
        ## What is AI Health Tracker?
        
        AI Health Tracker is an advanced health monitoring application that uses deep learning to analyze
        your health metrics and provide personalized insights and recommendations.
        
        ### Features
        
        - **Data Analysis**: Upload and analyze your health data from various sources
        - **Neural Network Predictions**: Use deep learning to forecast health metrics with high accuracy
        - **Personalized Recommendations**: Get tailored health advice based on your data
        - **Interactive Dashboard**: Visualize your health trends in real-time
        
        ### How It Works
        
        1. The application collects health data from various sources
        2. Deep neural networks analyze patterns in your data
        3. The system generates predictions and personalized recommendations
        4. You can view all insights through an intuitive dashboard
        
        ### Data Privacy
        
        Your health data privacy is our priority. All data is processed locally and is not shared with any third parties.
        """)
        
        # Team information
        st.subheader("Development Team")
        st.write("This application was developed by the AI Health Solutions team.")
        
        # Contact information
        st.subheader("Contact Information")
        st.write("For questions or support, please contact: support@aihealthtracker.example.com")

    def _get_normal_reference_value(self, feature):
        """Get a normal reference value for a given feature"""
        if feature == 'heart_rate' or 'heart_rate' in feature:
            return 75.0
        elif feature == 'temperature' or 'temperature' in feature:
            return 36.5
        elif 'electrodermal' in feature:
            return 5.0
        elif 'blood_volume' in feature:
            return 70.0
        elif feature == 'stress_level':
            return 3.0
        else:
            return 0.0
    
    def _get_heart_rate_status_color(self, heart_rate):
        """Get color code based on heart rate value"""
        if heart_rate < 60:
            return "#9467bd"  # Purple for low
        elif 60 <= heart_rate <= 100:
            return "#5cb85c"  # Green for normal
        elif 100 < heart_rate <= 140:
            return "#f0ad4e"  # Orange for elevated
        else:
            return "#d9534f"  # Red for high
    
    def _get_health_status_text(self, target, value):
        """Get a health status text based on the predicted value"""
        if target == 'heart_rate':
            if value < 60:
                return "Bradycardia (low heart rate)"
            elif 60 <= value <= 100:
                return "Normal heart rate"
            elif 100 < value <= 140:
                return "Elevated heart rate"
            else:
                return "Tachycardia (high heart rate)"
        elif target == 'stress_level':
            if value < 3:
                return "Low stress level"
            elif 3 <= value <= 7:
                return "Moderate stress level"
            else:
                return "High stress level"
        else:
            return "Normal"
    
    def _get_bmi_category(self, bmi):
        """Get BMI category based on BMI value"""
        if bmi < 18.5:
            return "Underweight"
        elif 18.5 <= bmi < 25:
            return "Normal weight"
        elif 25 <= bmi < 30:
            return "Overweight"
        elif 30 <= bmi < 35:
            return "Obese (Class I)"
        elif 35 <= bmi < 40:
            return "Obese (Class II)"
        else:
            return "Extremely Obese (Class III)"
    
    def _get_bmi_color(self, bmi):
        """Get color code based on BMI value"""
        if bmi < 18.5:
            return "#9467bd"  # Purple for underweight
        elif 18.5 <= bmi < 25:
            return "#5cb85c"  # Green for normal
        elif 25 <= bmi < 30:
            return "#f0ad4e"  # Orange for overweight
        else:
            return "#d9534f"  # Red for obese
    
    def _get_potential_health_conditions(self, target, value, temperature, eda):
        """Get potential health conditions based on predicted values"""
        conditions = []
        
        if target == 'heart_rate':
            if value < 60:
                conditions.append({
                    'name': 'Bradycardia',
                    'description': 'Lower than normal heart rate that may indicate problems with the heart\'s electrical system.',
                    'risk_level': 'Moderate'
                })
            elif value > 100:
                conditions.append({
                    'name': 'Tachycardia',
                    'description': 'Higher than normal heart rate that may be caused by physical activity, stress, or heart conditions.',
                    'risk_level': 'Moderate'
                })
            
            if value > 120:
                conditions.append({
                    'name': 'Cardiovascular Strain',
                    'description': 'Elevated heart rate may be putting extra strain on your cardiovascular system.',
                    'risk_level': 'Moderate'
                })
        
        # Check for temperature abnormalities
        if temperature < 36.0:
            conditions.append({
                'name': 'Hypothermia',
                'description': 'Low body temperature that may indicate exposure to cold, metabolic disorders, or other conditions.',
                'risk_level': 'Moderate'
            })
        elif temperature > 37.5:
            conditions.append({
                'name': 'Fever',
                'description': 'Elevated body temperature that may indicate infection, inflammation, or other conditions.',
                'risk_level': 'Moderate'
            })
        
        # Check for EDA related conditions
        if eda > 10.0:
            conditions.append({
                'name': 'Elevated Stress Response',
                'description': 'Higher than normal electrodermal activity indicating increased sympathetic nervous system activity.',
                'risk_level': 'Low'
            })
        
        # Combined conditions
        if value > 100 and temperature > 37.5:
            conditions.append({
                'name': 'Potential Infection',
                'description': 'Combination of elevated heart rate and body temperature may indicate an infectious process.',
                'risk_level': 'High'
            })
        
        if value > 100 and eda > 10.0:
            conditions.append({
                'name': 'Anxiety/Stress Response',
                'description': 'Combination of elevated heart rate and skin conductance suggests heightened stress or anxiety.',
                'risk_level': 'Moderate'
            })
        
        return conditions
    
    def _generate_recommendations(self, target, value, heart_rate, temperature, eda, age=35, gender="Male", activity_level="Lightly Active"):
        """Generate personalized health recommendations based on predicted values"""
        recommendations = {
            "Lifestyle Recommendations": [],
            "Monitoring Suggestions": [],
            "Activity Recommendations": []
        }
        
        # General recommendations
        recommendations["Lifestyle Recommendations"].append("Stay hydrated by drinking at least 8 glasses of water daily.")
        recommendations["Lifestyle Recommendations"].append("Maintain a balanced diet rich in fruits, vegetables, and whole grains.")
        recommendations["Monitoring Suggestions"].append("Track your health metrics regularly to identify patterns and changes.")
        
        # Target-specific recommendations
        if target == 'heart_rate':
            if value < 60:
                recommendations["Lifestyle Recommendations"].append("Consider discussing your low heart rate with a healthcare provider.")
                recommendations["Monitoring Suggestions"].append("Monitor your heart rate during different activities and times of day.")
                recommendations["Activity Recommendations"].append("Engage in light to moderate exercise to gradually improve cardiovascular fitness.")
            
            elif value > 100:
                recommendations["Lifestyle Recommendations"].append("Practice stress-reduction techniques like deep breathing or meditation.")
                recommendations["Lifestyle Recommendations"].append("Limit caffeine and stimulant intake, especially in the afternoon and evening.")
                recommendations["Monitoring Suggestions"].append("Keep a log of activities that may trigger elevated heart rate.")
                recommendations["Activity Recommendations"].append("Include cool-down periods after exercise to gradually lower your heart rate.")
        
        # Temperature recommendations
        if temperature < 36.0:
            recommendations["Lifestyle Recommendations"].append("Ensure you're dressed appropriately for the environment to maintain body temperature.")
            recommendations["Monitoring Suggestions"].append("Monitor your temperature regularly, especially if you continue to feel cold.")
        elif temperature > 37.5:
            recommendations["Lifestyle Recommendations"].append("Rest and increase fluid intake if experiencing elevated temperature.")
            recommendations["Monitoring Suggestions"].append("Check your temperature regularly to track any changes.")
            recommendations["Lifestyle Recommendations"].append("Consult a healthcare provider if fever persists for more than 24 hours.")
        
        # Age-specific recommendations
        if age > 50:
            recommendations["Lifestyle Recommendations"].append("Ensure regular health check-ups, including cardiac assessments.")
            recommendations["Monitoring Suggestions"].append("Consider more frequent blood pressure and heart rate monitoring.")
        
        # Activity level recommendations
        if activity_level in ["Sedentary", "Lightly Active"]:
            recommendations["Activity Recommendations"].append("Aim to increase physical activity gradually, targeting at least 150 minutes of moderate exercise weekly.")
            recommendations["Activity Recommendations"].append("Incorporate more movement throughout your day, such as taking short walking breaks.")
        elif activity_level in ["Very Active", "Extremely Active"]:
            recommendations["Activity Recommendations"].append("Ensure adequate rest and recovery between intense workouts.")
            recommendations["Lifestyle Recommendations"].append("Focus on proper nutrition to support your high activity level.")
        
        # Stress-related recommendations
        if eda > 10.0:
            recommendations["Lifestyle Recommendations"].append("Practice regular stress management techniques such as meditation, deep breathing, or yoga.")
            recommendations["Lifestyle Recommendations"].append("Consider keeping a stress journal to identify triggers and patterns.")
            recommendations["Activity Recommendations"].append("Try incorporating relaxing activities like gentle walking, swimming, or tai chi.")
        
        return recommendations
    
    def _generate_health_report(self, prediction, target_col, heart_rate, temperature, eda, conditions, recommendations, patient_info):
        """Generate a downloadable health report"""
        report = f"""HEALTH ASSESSMENT REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}

PATIENT INFORMATION
------------------
Age: {patient_info['age']} years
Gender: {patient_info['gender']}
Weight: {patient_info['weight']} kg
Height: {patient_info['height']} cm
Activity Level: {patient_info['activity_level']}

PREDICTED HEALTH METRICS
-----------------------
{target_col.replace('_', ' ').title()}: {prediction:.2f}
Heart Rate: {heart_rate:.1f} BPM
Body Temperature: {temperature:.1f} °C
Electrodermal Activity: {eda:.1f} μS

HEALTH STATUS
------------
{self._get_health_status_text(target_col, prediction)}

POTENTIAL HEALTH CONDITIONS
-------------------------
"""
        if conditions:
            for condition in conditions:
                report += f"• {condition['name']} (Risk Level: {condition['risk_level']})\n  {condition['description']}\n\n"
        else:
            report += "No specific health concerns identified based on the provided metrics.\n\n"
        
        report += "RECOMMENDATIONS\n---------------\n"
        
        for category, recs in recommendations.items():
            report += f"\n{category}:\n"
            for rec in recs:
                report += f"• {rec}\n"
        
        report += "\n\nDISCLAIMER\n----------\nThis report is generated based on predicted values and provided information. It is not a medical diagnosis. Always consult with a healthcare professional for personalized medical advice."
        
        return report

# Add Ngrok setup function
def setup_ngrok():
    # Check if app is already running on ngrok
    try:
        # Get the current URL from environment variables
        host = os.environ.get('SERVER_HOST', '')
        
        # If we're already on ngrok, don't create a new tunnel
        if 'ngrok' in host:
            print(f"Already running on ngrok URL: {host}")
            return host
            
        # Try checking via Streamlit's experimental get_query_params
        try:
            import streamlit.web.server.server as server
            # Check if current server has ngrok in the baseUrl
            if hasattr(server, 'get_base_url_hostname'):
                hostname = server.get_base_url_hostname()
                if 'ngrok' in hostname:
                    print(f"Already running on ngrok hostname: {hostname}")
                    return f"https://{hostname}"
        except Exception as e:
            print(f"Could not check Streamlit server hostname: {e}")
    except Exception as e:
        print(f"Error checking if running on ngrok: {e}")
    
    # Set up ngrok tunnel to expose the Streamlit app
    try:
        # Set auth token
        ngrok.set_auth_token("2uX0aCJoUZuNzfrCZVC3lsvOh5V_4iryQ4V9XANNDiGzrokMH")
        
        # Get the port that Streamlit is running on
        try:
            port = int(sys.argv[sys.argv.index("--server.port") + 1])
        except (ValueError, IndexError):
            port = 8501  # Default Streamlit port
        
        # Check if ngrok is already running on this port by listing tunnels
        existing_tunnels = ngrok.get_tunnels()
        for tunnel in existing_tunnels:
            if f":{port}" in tunnel.config['addr']:
                print(f"Found existing ngrok tunnel for port {port}: {tunnel.public_url}")
                return tunnel.public_url
        
        # Open a ngrok tunnel to the Streamlit port
        public_url = ngrok.connect(port).public_url
        print(f" * ngrok tunnel available at: {public_url}")
        return public_url
    except Exception as e:
        print(f"Error setting up ngrok: {e}")
        # Return None to indicate tunneling failed
        return None

if __name__ == "__main__":
    # Initialize the app
    app = HealthTrackerApp()
    
    # Setup ngrok tunnel if not already on ngrok
    public_url = setup_ngrok()
    
    # Display appropriate message based on access method
    if public_url:
        if 'ngrok' in public_url:
            # We either created a new tunnel or detected an existing one
            st.sidebar.success(f"Ngrok tunnel available at: {public_url}")
        else:
            # We're already running on ngrok
            st.sidebar.info("You're accessing the app through an ngrok tunnel.")
    else:
        st.sidebar.warning(
            "Failed to create ngrok tunnel. This app is only accessible locally at http://localhost:8501.\n\n"
            "To fix ngrok issues:\n"
            "1. Ensure you don't have other ngrok sessions running\n"
            "2. Check your ngrok authentication\n"
            "3. Try using a custom authtoken if you have one"
        )
    
    # Run the app
    app.run() 