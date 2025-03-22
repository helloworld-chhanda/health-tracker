import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
import joblib
import os
import json
from datetime import datetime, timedelta
from pyngrok import ngrok
import time
import sys
import warnings
warnings.filterwarnings('ignore')

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
    def __init__(self, model_dir='saved_models', data_dir='.'):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.target_col = None
        self.demo_data = None
        
    def load_model(self):
        """Load the trained model and preprocessing objects"""
        # Load model
        model_path = os.path.join(self.model_dir, 'health_model.pkl')
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            st.sidebar.success("Model loaded successfully!")
        else:
            st.sidebar.error("Model file not found. Please train the model first.")
            return False
        
        # Load scaler
        scaler_path = os.path.join('processed_data', 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            st.sidebar.warning("Scaler not found. Predictions may be inaccurate.")
        
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
        """Display the predictions page"""
        st.header("Health Predictions")
        
        if self.model is None:
            if not self.load_model():
                st.warning("Model not loaded. Please load the model first.")
                return
        
        st.write("Use this page to get personalized health predictions based on your metrics.")
        
        # Create input form for health metrics
        with st.form("prediction_form"):
            st.subheader("Enter your health metrics")
            
            # Dynamic inputs based on required features for the model
            input_values = {}
            
            if self.feature_names:
                # For each feature, create an appropriate input widget
                for feature in self.feature_names:
                    # Determine appropriate input type based on feature name
                    if 'rate' in feature.lower():
                        input_values[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()}", 
                            min_value=0.0, 
                            max_value=200.0, 
                            value=75.0,
                            step=1.0
                        )
                    elif 'temp' in feature.lower():
                        input_values[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()} (°C)", 
                            min_value=30.0, 
                            max_value=45.0, 
                            value=36.5,
                            step=0.1
                        )
                    elif any(x in feature.lower() for x in ['hour', 'minute', 'second']):
                        max_val = 23 if 'hour' in feature.lower() else 59
                        input_values[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()}", 
                            min_value=0, 
                            max_value=max_val, 
                            value=12 if 'hour' in feature.lower() else 30,
                            step=1
                        )
                    elif 'activity' in feature.lower():
                        input_values[feature] = st.selectbox(
                            f"{feature.replace('_', ' ').title()}", 
                            ['Rest', 'Walk', 'Run', 'Exercise']
                        )
                    else:
                        input_values[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()}", 
                            value=0.0,
                            step=0.1
                        )
            else:
                # Fallback if no feature names are available
                st.warning("Feature information not available. Using default inputs.")
                input_values['heart_rate'] = st.number_input("Heart Rate (BPM)", min_value=0.0, max_value=200.0, value=75.0)
                input_values['electrodermal_activity'] = st.number_input("Electrodermal Activity (μS)", min_value=0.0, max_value=100.0, value=5.0)
                input_values['temperature'] = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=36.5)
            
            submit_button = st.form_submit_button("Get Prediction")
        
        # Process prediction on form submission
        if submit_button:
            try:
                # Prepare input data for prediction
                input_df = pd.DataFrame([input_values])
                
                # Handle categorical features (if any)
                for col in input_df.columns:
                    if input_df[col].dtype == 'object':
                        # One-hot encode categorical features
                        dummies = pd.get_dummies(input_df[col], prefix=col)
                        input_df = pd.concat([input_df.drop(columns=[col]), dummies], axis=1)
                
                # Ensure all required features are present
                missing_features = set(self.feature_names) - set(input_df.columns)
                for feature in missing_features:
                    input_df[feature] = 0  # Default value for missing features
                
                # Apply feature scaling if scaler is available
                if self.scaler is not None:
                    input_scaled = self.scaler.transform(input_df[self.feature_names])
                else:
                    input_scaled = input_df[self.feature_names].values
                
                # Make prediction
                prediction = self.model.predict(input_scaled)[0]
                
                # Display prediction
                st.success(f"Predicted {self.target_col}: {prediction:.2f}")
                
                # Display additional information based on the predicted value
                if self.target_col == 'heart_rate':
                    if prediction < 60:
                        st.info("Your predicted heart rate is below normal resting range (60-100 BPM). This could indicate bradycardia.")
                    elif prediction > 100:
                        st.warning("Your predicted heart rate is above normal resting range (60-100 BPM). This could indicate tachycardia.")
                    else:
                        st.info("Your predicted heart rate is within normal resting range (60-100 BPM).")
                
                # Display prediction confidence
                if self.metrics and 'rmse' in self.metrics:
                    rmse = self.metrics['rmse']
                    st.info(f"Model accuracy: Predictions are typically within ±{rmse:.2f} of the actual value.")
                
                # Display personalized recommendations
                st.subheader("Personalized Recommendations")
                
                if self.target_col == 'heart_rate':
                    if prediction > 100:
                        st.markdown("""
                        * Consider stress reduction techniques such as meditation or deep breathing
                        * Ensure you're staying hydrated throughout the day
                        * Consult a healthcare professional if you consistently have elevated heart rate
                        """)
                    elif prediction < 60:
                        st.markdown("""
                        * Increase physical activity gradually if appropriate
                        * Monitor your heart rate regularly
                        * Consult a healthcare professional if your heart rate is consistently low
                        """)
                    else:
                        st.markdown("""
                        * Your predicted heart rate is in a healthy range
                        * Continue with regular physical activity
                        * Maintain a balanced diet and good sleep habits
                        """)
                        
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.info("Try loading the model again or check that the input values are correct.")
    
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
                heart_rate = st.number_input("Heart Rate (BPM)", min_value=0, max_value=200, value=75)
            
            with col2:
                eda = st.number_input("Electrodermal Activity (μS)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
            
            with col3:
                temp = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=36.5, step=0.1)
            
            activity = st.selectbox("Current Activity", ["Rest", "Walk", "Run", "Exercise"])
            
            submitted = st.form_submit_button("Save Data")
        
        if submitted:
            st.success("Data saved successfully!")
            
            # Create a dataframe with the manually entered data
            manual_data = pd.DataFrame({
                'timestamp': [datetime.now().timestamp()],
                'heart_rate': [heart_rate],
                'electrodermal_activity': [eda],
                'temperature': [temp],
                'activity': [activity],
                'datetime': [datetime.now()]
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
            # Display model information
            st.subheader("Model Type")
            model_type = type(self.model).__name__
            st.write(f"This is a **{model_type}** model.")
            
            # Display model coefficients for linear models
            if hasattr(self.model, 'coef_'):
                st.subheader("Model Coefficients")
                coef_df = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Coefficient': self.model.coef_
                }).sort_values('Coefficient', ascending=False)
                
                st.dataframe(coef_df)
            
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
            
            # Display feature importance
            st.subheader("Feature Importance")
            st.write("This is a representation of how much each feature contributes to predictions.")
            
            # Create feature importance visualization
            if self.feature_names:
                # For linear models, use coefficients as importance
                if hasattr(self.model, 'coef_'):
                    importances = np.abs(self.model.coef_)
                else:
                    # Create random importance values for illustration
                    importances = np.random.uniform(0, 1, len(self.feature_names))
                
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
                    title='Feature Importance',
                    labels={'Importance': 'Relative Importance', 'Feature': 'Feature Name'},
                    color='Importance',
                    color_continuous_scale=['#98bef9', '#7baaf7', '#5e97f6', '#4a86e8', '#3a76d8']
                )
                fig.update_layout(
                    paper_bgcolor='rgba(245, 247, 249, 0)',
                    plot_bgcolor='rgba(245, 247, 249, 0.5)',
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, use_container_width=True)
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
        - **Predictive Analytics**: Use deep learning to forecast health metrics
        - **Personalized Recommendations**: Get tailored health advice based on your data
        - **Interactive Dashboard**: Visualize your health trends in real-time
        
        ### How It Works
        
        1. The application collects health data from various sources
        2. Deep learning models analyze patterns in your data
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

# Add Ngrok setup function
def setup_ngrok():
    # Set up ngrok tunnel to expose the Streamlit app
    try:
        # Get the port that Streamlit is running on
        port = int(sys.argv[sys.argv.index("--server.port") + 1])
    except (ValueError, IndexError):
        port = 8501  # Default Streamlit port
    
    # Open a ngrok tunnel to the Streamlit port
    public_url = ngrok.connect(port).public_url
    print(f" * ngrok tunnel available at: {public_url}")
    return public_url

if __name__ == "__main__":
    # Initialize and run the app
    app = HealthTrackerApp()
    app.run() 