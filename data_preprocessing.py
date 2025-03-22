import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

class HealthDataProcessor:
    def __init__(self, data_dir='.'):
        self.data_dir = data_dir
        self.hr_data = None
        self.eda_data = None
        self.temp_data = None
        self.bvp_data = None
        self.merged_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
    
    def load_data(self):
        """Load data from CSV files"""
        # Load HR data
        hr_path = os.path.join(self.data_dir, 'HR.csv')
        if os.path.exists(hr_path):
            hr_df = pd.read_csv(hr_path, header=None)
            timestamp = float(hr_df.iloc[0, 0])
            sample_rate = float(hr_df.iloc[1, 0])
            hr_values = hr_df.iloc[2:].values.flatten()
            # Create proper time index
            times = np.arange(len(hr_values)) / sample_rate
            self.hr_data = pd.DataFrame({
                'timestamp': times + timestamp,
                'heart_rate': hr_values
            })
            print(f"Loaded HR data: {len(self.hr_data)} records")
        
        # Load EDA data
        eda_path = os.path.join(self.data_dir, 'EDA.csv')
        if os.path.exists(eda_path):
            eda_df = pd.read_csv(eda_path, header=None)
            timestamp = float(eda_df.iloc[0, 0])
            sample_rate = float(eda_df.iloc[1, 0])
            eda_values = eda_df.iloc[2:].values.flatten()
            # Create proper time index
            times = np.arange(len(eda_values)) / sample_rate
            self.eda_data = pd.DataFrame({
                'timestamp': times + timestamp,
                'electrodermal_activity': eda_values
            })
            print(f"Loaded EDA data: {len(self.eda_data)} records")
            
        # Load TEMP data
        temp_path = os.path.join(self.data_dir, 'TEMP.csv')
        if os.path.exists(temp_path):
            temp_df = pd.read_csv(temp_path, header=None)
            timestamp = float(temp_df.iloc[0, 0])
            sample_rate = float(temp_df.iloc[1, 0])
            temp_values = temp_df.iloc[2:].values.flatten()
            # Create proper time index
            times = np.arange(len(temp_values)) / sample_rate
            self.temp_data = pd.DataFrame({
                'timestamp': times + timestamp,
                'temperature': temp_values
            })
            print(f"Loaded TEMP data: {len(self.temp_data)} records")
            
        # Load BVP data if it's not too large
        try:
            bvp_path = os.path.join(self.data_dir, 'BVP.csv')
            if os.path.exists(bvp_path):
                # Read just the header to get timestamp and sample rate
                bvp_df_header = pd.read_csv(bvp_path, header=None, nrows=2)
                timestamp = float(bvp_df_header.iloc[0, 0])
                sample_rate = float(bvp_df_header.iloc[1, 0])
                
                # Read the BVP data in chunks to handle large file size
                chunk_size = 10000
                chunks = []
                for chunk in pd.read_csv(bvp_path, header=None, skiprows=2, chunksize=chunk_size):
                    chunks.append(chunk)
                
                # Concatenate all chunks
                bvp_values = pd.concat(chunks).values.flatten()
                # Create proper time index
                times = np.arange(len(bvp_values)) / sample_rate
                self.bvp_data = pd.DataFrame({
                    'timestamp': times + timestamp,
                    'blood_volume_pulse': bvp_values
                })
                print(f"Loaded BVP data: {len(self.bvp_data)} records")
        except Exception as e:
            print(f"Could not load BVP data: {e}")
        
        return self
    
    def merge_data(self):
        """Merge different data sources based on timestamp"""
        dataframes = []
        if self.hr_data is not None:
            dataframes.append(self.hr_data)
        if self.eda_data is not None:
            dataframes.append(self.eda_data)
        if self.temp_data is not None:
            dataframes.append(self.temp_data)
        
        if not dataframes:
            raise ValueError("No data loaded to merge")
        
        # Start with the first dataframe
        merged = dataframes[0]
        
        # Merge with other dataframes one by one using timestamp as key
        for df in dataframes[1:]:
            # Resample to a common frequency before merging
            merged = pd.merge_asof(
                merged.sort_values('timestamp'),
                df.sort_values('timestamp'),
                on='timestamp',
                direction='nearest'
            )
        
        self.merged_data = merged
        print(f"Merged data shape: {self.merged_data.shape}")
        return self
    
    def clean_data(self):
        """Clean the data by handling missing values and outliers"""
        if self.merged_data is None:
            raise ValueError("No data to clean. Load and merge data first.")
        
        # Check for NaN values
        print(f"Missing values before imputation:\n{self.merged_data.isnull().sum()}")
        
        # Create a copy of the data for cleaning
        cleaned_data = self.merged_data.copy()
        
        # Remove rows where all feature values are NaN
        cleaned_data = cleaned_data.dropna(how='all')
        
        # Impute remaining NaN values with median for numerical features
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='median')
        
        # Skip the timestamp column for imputation
        numeric_columns = [col for col in numeric_columns if col != 'timestamp']
        
        if numeric_columns:
            cleaned_data[numeric_columns] = imputer.fit_transform(cleaned_data[numeric_columns])
        
        # Handle outliers using IQR method
        for column in numeric_columns:
            if column != 'timestamp':
                Q1 = cleaned_data[column].quantile(0.25)
                Q3 = cleaned_data[column].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define bounds for outliers
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                cleaned_data[column] = np.where(
                    cleaned_data[column] < lower_bound,
                    lower_bound,
                    np.where(
                        cleaned_data[column] > upper_bound,
                        upper_bound,
                        cleaned_data[column]
                    )
                )
        
        self.merged_data = cleaned_data
        print(f"Data after cleaning, shape: {self.merged_data.shape}")
        print(f"Missing values after imputation:\n{self.merged_data.isnull().sum()}")
        
        return self
    
    def create_features(self):
        """Create additional features from existing data"""
        if self.merged_data is None:
            raise ValueError("No data to create features from. Load and clean data first.")
        
        # Create a copy of the data
        df = self.merged_data.copy()
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Extract time-based features
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df['second'] = df['datetime'].dt.second
        df['day_of_week'] = df['datetime'].dt.dayofweek
        
        # Calculate rolling statistics if we have enough data
        if 'heart_rate' in df.columns and len(df) > 10:
            df['hr_rolling_mean'] = df['heart_rate'].rolling(window=10, min_periods=1).mean()
            df['hr_rolling_std'] = df['heart_rate'].rolling(window=10, min_periods=1).std()
        
        if 'electrodermal_activity' in df.columns and len(df) > 10:
            df['eda_rolling_mean'] = df['electrodermal_activity'].rolling(window=10, min_periods=1).mean()
            df['eda_rolling_std'] = df['electrodermal_activity'].rolling(window=10, min_periods=1).std()
        
        if 'temperature' in df.columns and len(df) > 10:
            df['temp_rolling_mean'] = df['temperature'].rolling(window=10, min_periods=1).mean()
            df['temp_rolling_std'] = df['temperature'].rolling(window=10, min_periods=1).std()
        
        # Drop original datetime column
        df = df.drop(columns=['datetime'])
        
        # Calculate correlation with heart rate
        if 'heart_rate' in df.columns:
            corr_columns = [col for col in df.columns if col != 'timestamp' and col != 'heart_rate']
            if corr_columns:
                print("Correlations with heart rate:")
                for col in corr_columns:
                    correlation = df['heart_rate'].corr(df[col])
                    print(f"{col}: {correlation}")
        
        self.merged_data = df
        print(f"Data after feature creation, shape: {self.merged_data.shape}")
        
        return self
    
    def visualize_data(self, output_dir='visualizations'):
        """Generate visualizations for the data"""
        if self.merged_data is None:
            raise ValueError("No data to visualize. Load and process data first.")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot histograms for each feature
        feature_columns = [col for col in self.merged_data.columns if col != 'timestamp']
        
        for column in feature_columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.merged_data[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.savefig(os.path.join(output_dir, f'{column}_distribution.png'))
            plt.close()
        
        # Plot time series for important features
        important_features = [col for col in ['heart_rate', 'electrodermal_activity', 'temperature'] 
                              if col in self.merged_data.columns]
        
        if important_features:
            plt.figure(figsize=(12, 8))
            for feature in important_features:
                plt.plot(self.merged_data['timestamp'], self.merged_data[feature], label=feature)
            plt.title('Time Series of Health Metrics')
            plt.xlabel('Timestamp')
            plt.ylabel('Value')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'time_series.png'))
            plt.close()
        
        # Plot correlation heatmap
        numeric_columns = self.merged_data.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'timestamp']
        
        if len(numeric_columns) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.merged_data[numeric_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Matrix of Health Metrics')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
            plt.close()
        
        return self
    
    def prepare_for_modeling(self, target_col='heart_rate', test_size=0.2, random_state=42):
        """Prepare data for modeling by scaling and splitting into train/test sets"""
        if self.merged_data is None:
            raise ValueError("No data to prepare. Load and process data first.")
        
        # Make sure the target column exists
        if target_col not in self.merged_data.columns:
            available_columns = [col for col in self.merged_data.columns if col != 'timestamp']
            if not available_columns:
                raise ValueError("No suitable columns for target variable")
            target_col = available_columns[0]
            print(f"Target column '{target_col}' not found. Using '{target_col}' instead.")
        
        # Select features and target
        X = self.merged_data.drop(columns=['timestamp', target_col])
        y = self.merged_data[target_col]
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        self.target_col = target_col
        
        print(f"Data prepared for modeling:")
        print(f"X_train shape: {self.X_train.shape}")
        print(f"X_test shape: {self.X_test.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        print(f"y_test shape: {self.y_test.shape}")
        
        return self
    
    def save_processed_data(self, output_dir='processed_data'):
        """Save the processed data and preprocessing objects"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("No processed data to save. Prepare data for modeling first.")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the training and test data
        np.save(os.path.join(output_dir, 'X_train.npy'), self.X_train_scaled)
        np.save(os.path.join(output_dir, 'X_test.npy'), self.X_test_scaled)
        np.save(os.path.join(output_dir, 'y_train.npy'), self.y_train)
        np.save(os.path.join(output_dir, 'y_test.npy'), self.y_test)
        
        # Save the scaler
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        
        # Save feature names and target column
        pd.Series(self.feature_names).to_csv(os.path.join(output_dir, 'feature_names.csv'), index=False)
        pd.Series([self.target_col]).to_csv(os.path.join(output_dir, 'target_col.csv'), index=False)
        
        print(f"Processed data and preprocessing objects saved to {output_dir}")
        
        return self

# Usage example
if __name__ == "__main__":
    # Create data processor
    processor = HealthDataProcessor()
    
    # Execute data processing pipeline
    (processor
     .load_data()
     .merge_data()
     .clean_data()
     .create_features()
     .visualize_data()
     .prepare_for_modeling()
     .save_processed_data()
    ) 