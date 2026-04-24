import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Data loading and preprocessing utilities with categorical handling"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.letter_encoder = LabelEncoder()  # Specifically for Letter column
        self.columns = ['Letter', 'x_box', 'y_box', 'width', 'height', 'onpix_total', 
                       'x_bar', 'y_bar', 'x2_bar', 'y2_bar', 'xy_bar', 'x2y_bar', 
                       'xy2_bar', 'x_ege', 'xegvy', 'y_ege', 'yegvx']
        self.numeric_columns = [col for col in self.columns if col != 'Letter']
        self.categorical_columns = ['Letter']
        
    def load_data(self, filepath):
        """Load dataset from CSV file with proper type handling"""
        try:
            # Load data with all columns as string initially to handle mixed types
            data = pd.read_csv(filepath, dtype=str)
            
            # Convert numeric columns to float, handle errors
            for col in self.numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Check for missing columns
            missing_cols = [col for col in self.columns if col not in data.columns]
            if missing_cols:
                print(f"Warning: Missing columns: {missing_cols}")
                # Use available columns
                available_cols = [col for col in self.columns if col in data.columns]
                X_data_raw = data[available_cols]
            else:
                X_data_raw = data[self.columns]
            
            # Handle categorical columns (Letter)
            X_processed = []
            for col in self.columns:
                if col in X_data_raw.columns:
                    if col == 'Letter':
                        # Encode letter column
                        encoded = self.letter_encoder.fit_transform(X_data_raw[col].fillna('unknown'))
                        X_processed.append(encoded.reshape(-1, 1))
                    else:
                        # Numeric columns
                        values = pd.to_numeric(X_data_raw[col], errors='coerce').fillna(0).values
                        X_processed.append(values.reshape(-1, 1))
            
            # Combine all features
            X_data = np.hstack(X_processed)
            
            # Handle target column
            if 'Label' in data.columns or 'label' in data.columns:
                target_col = 'Label' if 'Label' in data.columns else 'label'
                y_data = data[target_col].values
                y_encoded = self.label_encoder.fit_transform(y_data)
            else:
                # Create synthetic labels based on patterns
                print("No label column found. Creating synthetic labels...")
                # Use simple heuristic: if features are extreme, mark as attack
                feature_means = np.mean(X_data, axis=1)
                threshold = np.percentile(feature_means, 70)
                y_data = np.where(feature_means > threshold, 'attack', 'normal')
                y_encoded = self.label_encoder.fit_transform(y_data)
            
            return data, X_data, y_data, y_encoded
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def preprocess_data(self, X, y=None, fit_scaler=True):
        """Preprocess features"""
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        if y is not None:
            return X_scaled, y
        return X_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test
    
    def create_sample_dataset(self, n_samples=1000, save_path="data/sample_data.csv"):
        """Create a sample dataset for testing with categorical data"""
        import os
        os.makedirs("data", exist_ok=True)
        
        np.random.seed(42)
        
        # Generate data
        data = {}
        
        # Letter column (categorical)
        letters = ['T', 'I', 'D', 'A', 'B', 'C', 'E', 'F', 'G', 'H']
        data['Letter'] = np.random.choice(letters, n_samples)
        
        # Numeric columns with different distributions for normal vs attack
        n_normal = int(n_samples * 0.7)
        n_attack = n_samples - n_normal
        
        for col in self.numeric_columns:
            # Normal traffic (low values)
            normal_vals = np.random.randn(n_normal) * 2 + 5
            # Attack traffic (higher, more spread out)
            attack_vals = np.random.randn(n_attack) * 5 + 10
            
            data[col] = np.concatenate([normal_vals, attack_vals])
        
        # Create labels
        labels = ['normal'] * n_normal + ['attack'] * n_attack
        data['Label'] = labels
        
        # Create DataFrame and shuffle
        df = pd.DataFrame(data)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        print(f"Sample dataset created at {save_path}")
        print(f"Total samples: {len(df)}")
        print(f"Normal samples: {n_normal}")
        print(f"Attack samples: {n_attack}")
        
        return df
    
    def encode_single_packet(self, packet_features_dict):
        """Encode a single packet for prediction"""
        features = []
        
        for col in self.columns:
            if col in packet_features_dict:
                value = packet_features_dict[col]
                if col == 'Letter':
                    # Encode letter
                    if hasattr(self.letter_encoder, 'classes_'):
                        # Transform using fitted encoder
                        try:
                            encoded = self.letter_encoder.transform([value])[0]
                        except:
                            # Use default encoding if not seen before
                            encoded = 0
                    else:
                        encoded = 0
                    features.append(encoded)
                else:
                    # Numeric value
                    try:
                        features.append(float(value))
                    except:
                        features.append(0.0)
            else:
                # Missing column, use 0
                features.append(0.0)
        
        return np.array(features).reshape(1, -1)