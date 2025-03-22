import os
import subprocess
import time
import argparse
from data_preprocessing import HealthDataProcessor
from model import HealthModelBuilder
import threading
import sys

def run_data_preprocessing():
    """Run data preprocessing pipeline"""
    print("Starting data preprocessing...")
    
    # Create processor
    processor = HealthDataProcessor()
    
    # Execute data processing pipeline
    try:
        (processor
         .load_data()
         .merge_data()
         .clean_data()
         .create_features()
         .visualize_data()
         .prepare_for_modeling()
         .save_processed_data()
        )
        print("Data preprocessing completed successfully!")
        return True
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        return False

def run_model_training():
    """Run model training pipeline"""
    print("Starting model training...")
    
    # Create model builder
    model_builder = HealthModelBuilder()
    
    # Execute model building pipeline
    try:
        (model_builder
         .load_processed_data()
         .tune_hyperparameters(cv=3, n_iter=5)  # Reduced iterations for faster execution
         .build_model()
         .train_model()
         .evaluate_model()
         .plot_training_history()
         .save_model()
        )
        print("Model training completed successfully!")
        return True
    except Exception as e:
        print(f"Error during model training: {e}")
        return False

def run_streamlit_app(use_ngrok=False):
    """Run Streamlit app"""
    print("Starting Streamlit app...")
    
    if use_ngrok:
        cmd = ["streamlit", "run", "app.py", "--server.headless", "true"]
    else:
        cmd = ["streamlit", "run", "app.py"]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Give Streamlit time to start up
        time.sleep(5)
        
        if use_ngrok:
            from pyngrok import ngrok
            
            # Find the port that Streamlit is running on
            port = 8501  # Default Streamlit port
            
            # Open a ngrok tunnel to the Streamlit port
            public_url = ngrok.connect(port).public_url
            print(f"Ngrok tunnel opened at: {public_url}")
            print(f"Use this URL to access the Streamlit app from anywhere!")
        
        print("Streamlit app is running. Press Ctrl+C to stop.")
        
        # Keep the app running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping Streamlit app...")
            process.terminate()
            process.wait()
            if use_ngrok:
                ngrok.kill()
    
    except Exception as e:
        print(f"Error running Streamlit app: {e}")
        return False
    
    return True

def run_full_pipeline(preprocess=True, train=True, run_app=True, use_ngrok=False):
    """Run the full pipeline"""
    # Run data preprocessing if requested
    if preprocess:
        success = run_data_preprocessing()
        if not success:
            print("Data preprocessing failed. Stopping pipeline.")
            return False
    
    # Run model training if requested
    if train:
        success = run_model_training()
        if not success:
            print("Model training failed. Stopping pipeline.")
            return False
    
    # Run Streamlit app if requested
    if run_app:
        success = run_streamlit_app(use_ngrok)
        if not success:
            print("Streamlit app failed to start.")
            return False
    
    return True

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Health Tracker Pipeline")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip data preprocessing")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--skip-app", action="store_true", help="Skip running Streamlit app")
    parser.add_argument("--use-ngrok", action="store_true", help="Use ngrok to expose Streamlit app")
    args = parser.parse_args()
    
    # Run the full pipeline with requested options
    run_full_pipeline(
        preprocess=not args.skip_preprocess,
        train=not args.skip_train,
        run_app=not args.skip_app,
        use_ngrok=args.use_ngrok
    ) 