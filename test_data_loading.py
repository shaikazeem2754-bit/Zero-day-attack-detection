#!/usr/bin/env python3
"""
Test data loading with 10,000 rows of realistic intrusion detection data
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_processor import DataProcessor

def generate_large_dataset(n_samples=10000):
    """Generate a large dataset with 10,000 rows"""
    
    print(f"Generating {n_samples} rows of test data...")
    
    np.random.seed(42)  # For reproducibility
    
    # Letter categories (more variety for realistic data)
    letters = ['T', 'I', 'D', 'A', 'B', 'C', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']
    
    # Define patterns for different attack types
    attack_patterns = {
        'normal': {
            'scale': 1.0,
            'shift': 0,
            'letters': ['T', 'I', 'D', 'A', 'B', 'C', 'E'],
            'probability': 0.7
        },
        'ddos_attack': {
            'scale': 3.0,
            'shift': 5,
            'letters': ['F', 'G', 'H', 'J', 'K'],
            'probability': 0.1
        },
        'port_scan': {
            'scale': 2.5,
            'shift': 3,
            'letters': ['L', 'M', 'N', 'O'],
            'probability': 0.1
        },
        'malware': {
            'scale': 4.0,
            'shift': 7,
            'letters': ['P', 'Q', 'R', 'S'],
            'probability': 0.1
        }
    }
    
    # Prepare data containers
    data = {
        'Letter': [],
        'x_box': [],
        'y_box': [],
        'width': [],
        'height': [],
        'onpix_total': [],
        'x_bar': [],
        'y_bar': [],
        'x2_bar': [],
        'y2_bar': [],
        'xy_bar': [],
        'x2y_bar': [],
        'xy2_bar': [],
        'x_ege': [],
        'xegvy': [],
        'y_ege': [],
        'yegvx': [],
        'Label': []
    }
    
    # Generate data with different patterns
    for i in range(n_samples):
        # Determine attack type based on probability
        rand = np.random.random()
        cumulative_prob = 0
        attack_type = 'normal'
        
        for atype, pattern in attack_patterns.items():
            cumulative_prob += pattern['probability']
            if rand <= cumulative_prob:
                attack_type = atype
                break
        
        pattern = attack_patterns[attack_type]
        
        # Generate letter based on attack type
        letter = np.random.choice(pattern['letters'])
        data['Letter'].append(letter)
        
        # Generate numeric features with pattern-specific characteristics
        # Base values with noise
        base_noise = np.random.randn(16) * pattern['scale']
        shift = pattern['shift']
        
        # Different features have different ranges
        features = {
            'x_box': np.random.normal(3 + shift, 1 + pattern['scale']/2),
            'y_box': np.random.normal(8 + shift, 2 + pattern['scale']/2),
            'width': np.random.normal(4 + shift/2, 1 + pattern['scale']/3),
            'height': np.random.normal(6 + shift/2, 1.5 + pattern['scale']/3),
            'onpix_total': np.random.normal(3 + shift, 2 + pattern['scale']),
            'x_bar': np.random.normal(8 + shift, 2 + pattern['scale']),
            'y_bar': np.random.normal(7 + shift, 2 + pattern['scale']),
            'x2_bar': np.random.normal(4 + shift/2, 1.5 + pattern['scale']/2),
            'y2_bar': np.random.normal(6 + shift/2, 2 + pattern['scale']/2),
            'xy_bar': np.random.normal(8 + shift, 2.5 + pattern['scale']),
            'x2y_bar': np.random.normal(6 + shift/2, 2 + pattern['scale']/2),
            'xy2_bar': np.random.normal(7 + shift/2, 2 + pattern['scale']/2),
            'x_ege': np.random.normal(2 + shift/3, 1 + pattern['scale']/3),
            'xegvy': np.random.normal(6 + shift/2, 1.5 + pattern['scale']/2),
            'y_ege': np.random.normal(2 + shift/3, 1 + pattern['scale']/3),
            'yegvx': np.random.normal(7 + shift/2, 1.5 + pattern['scale']/2)
        }
        
        # Add some correlation between features for realism
        if attack_type != 'normal':
            # Attacks show more correlation between certain features
            features['x_bar'] += features['y_bar'] * 0.3
            features['xy_bar'] += features['x2y_bar'] * 0.2
        
        # Ensure non-negative values (most features should be positive)
        for key in features:
            features[key] = abs(features[key])
        
        # Add to data dictionary
        for key, value in features.items():
            data[key].append(value)
        
        # Add label
        data['Label'].append(attack_type)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{n_samples} rows...")
    
    return pd.DataFrame(data)

def create_balanced_dataset(n_samples=10000):
    """Create a balanced dataset with equal distribution of attack types"""
    
    print(f"\nCreating balanced dataset with {n_samples} rows...")
    
    # Calculate samples per class
    n_classes = 4  # normal, ddos_attack, port_scan, malware
    samples_per_class = n_samples // n_classes
    
    np.random.seed(42)
    
    # Letters for each class
    class_letters = {
        'normal': ['T', 'I', 'D', 'A', 'B', 'C', 'E'],
        'ddos_attack': ['F', 'G', 'H', 'J', 'K'],
        'port_scan': ['L', 'M', 'N', 'O'],
        'malware': ['P', 'Q', 'R', 'S']
    }
    
    # Parameters for each class
    class_params = {
        'normal': {'scale': 1.0, 'shift': 0},
        'ddos_attack': {'scale': 3.5, 'shift': 6},
        'port_scan': {'scale': 2.8, 'shift': 4},
        'malware': {'scale': 4.2, 'shift': 8}
    }
    
    data = {
        'Letter': [],
        'x_box': [],
        'y_box': [],
        'width': [],
        'height': [],
        'onpix_total': [],
        'x_bar': [],
        'y_bar': [],
        'x2_bar': [],
        'y2_bar': [],
        'xy_bar': [],
        'x2y_bar': [],
        'xy2_bar': [],
        'x_ege': [],
        'xegvy': [],
        'y_ege': [],
        'yegvx': [],
        'Label': []
    }
    
    # Generate balanced data
    for class_name in class_params.keys():
        print(f"  Generating {samples_per_class} samples for {class_name}...")
        params = class_params[class_name]
        letters = class_letters[class_name]
        
        for i in range(samples_per_class):
            # Letter
            letter = np.random.choice(letters)
            data['Letter'].append(letter)
            
            # Generate features
            scale = params['scale']
            shift = params['shift']
            
            # Base features with class-specific patterns
            features = {
                'x_box': abs(np.random.normal(3 + shift/2, 1 + scale/2)),
                'y_box': abs(np.random.normal(8 + shift/2, 2 + scale/2)),
                'width': abs(np.random.normal(4 + shift/3, 1 + scale/3)),
                'height': abs(np.random.normal(6 + shift/3, 1.5 + scale/3)),
                'onpix_total': abs(np.random.normal(3 + shift, 2 + scale)),
                'x_bar': abs(np.random.normal(8 + shift, 2 + scale)),
                'y_bar': abs(np.random.normal(7 + shift, 2 + scale)),
                'x2_bar': abs(np.random.normal(4 + shift/2, 1.5 + scale/2)),
                'y2_bar': abs(np.random.normal(6 + shift/2, 2 + scale/2)),
                'xy_bar': abs(np.random.normal(8 + shift, 2.5 + scale)),
                'x2y_bar': abs(np.random.normal(6 + shift/2, 2 + scale/2)),
                'xy2_bar': abs(np.random.normal(7 + shift/2, 2 + scale/2)),
                'x_ege': abs(np.random.normal(2 + shift/3, 1 + scale/3)),
                'xegvy': abs(np.random.normal(6 + shift/2, 1.5 + scale/2)),
                'y_ege': abs(np.random.normal(2 + shift/3, 1 + scale/3)),
                'yegvx': abs(np.random.normal(7 + shift/2, 1.5 + scale/2))
            }
            
            # Add class-specific patterns
            if class_name == 'ddos_attack':
                features['x_bar'] += features['y_bar'] * 0.4
                features['width'] *= 1.5
            elif class_name == 'port_scan':
                features['height'] *= 1.3
                features['x2y_bar'] += 2
            elif class_name == 'malware':
                features['onpix_total'] *= 2
                features['xy_bar'] += 3
            
            # Add to data
            for key, value in features.items():
                data[key].append(value)
            
            data['Label'].append(class_name)
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def test_data_loading_with_large_dataset():
    """Test data loading with 10,000 rows"""
    
    print("="*80)
    print("INTRUSION DETECTION SYSTEM - DATA LOADING TEST")
    print("="*80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create directory if not exists
    os.makedirs("data", exist_ok=True)
    
    # Option 1: Generate dataset with natural distribution (more normal traffic)
    print("Option 1: Generating dataset with natural distribution (70% normal, 30% attacks)...")
    df_natural = generate_large_dataset(10000)
    natural_path = "data/natural_distribution_10000.csv"
    df_natural.to_csv(natural_path, index=False)
    print(f"  ✓ Saved to: {natural_path}")
    
    # Option 2: Generate balanced dataset for better training
    print("\nOption 2: Generating balanced dataset (equal distribution of attack types)...")
    df_balanced = create_balanced_dataset(10000)
    balanced_path = "data/balanced_distribution_10000.csv"
    df_balanced.to_csv(balanced_path, index=False)
    print(f"  ✓ Saved to: {balanced_path}")
    
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    # Display statistics for natural distribution
    print("\n1. NATURAL DISTRIBUTION DATASET:")
    print("-"*40)
    print(f"Total samples: {len(df_natural)}")
    print(f"Features: {len(df_natural.columns) - 1}")  # Excluding Label
    print(f"\nClass Distribution:")
    class_counts = df_natural['Label'].value_counts()
    for label, count in class_counts.items():
        percentage = (count / len(df_natural)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    print(f"\nFeature Statistics (first 5 features):")
    for col in ['x_box', 'y_box', 'width', 'height', 'onpix_total']:
        print(f"  {col}: mean={df_natural[col].mean():.2f}, std={df_natural[col].std():.2f}, "
              f"min={df_natural[col].min():.2f}, max={df_natural[col].max():.2f}")
    
    # Display statistics for balanced distribution
    print("\n2. BALANCED DISTRIBUTION DATASET:")
    print("-"*40)
    print(f"Total samples: {len(df_balanced)}")
    print(f"Features: {len(df_balanced.columns) - 1}")
    print(f"\nClass Distribution:")
    class_counts_bal = df_balanced['Label'].value_counts()
    for label, count in class_counts_bal.items():
        percentage = (count / len(df_balanced)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Test data loading with DataProcessor
    print("\n" + "="*80)
    print("TESTING DATA PROCESSOR WITH LARGE DATASET")
    print("="*80)
    
    processor = DataProcessor()
    
    try:
        print("\nLoading natural distribution dataset...")
        start_time = datetime.now()
        data, X, y, y_encoded = processor.load_data(natural_path)
        load_time = (datetime.now() - start_time).total_seconds()
        
        print(f"  ✓ Data loaded successfully in {load_time:.2f} seconds")
        print(f"  Data shape: {data.shape}")
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Labels shape: {y.shape}")
        print(f"  Unique labels: {processor.label_encoder.classes_}")
        
        # Test preprocessing
        print("\nPreprocessing data...")
        start_time = datetime.now()
        X_scaled = processor.preprocess_data(X, fit_scaler=True)
        preprocess_time = (datetime.now() - start_time).total_seconds()
        print(f"  ✓ Data preprocessed in {preprocess_time:.2f} seconds")
        print(f"  Scaled feature shape: {X_scaled.shape}")
        print(f"  Feature range after scaling: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
        
        # Test train-test split
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = processor.split_data(X_scaled, y_encoded)
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        
        # Test encoding a single packet
        print("\nTesting single packet encoding...")
        sample_packet = {
            'Letter': 'T',
            'x_box': '2.5',
            'y_box': '8.3',
            'width': '3.1',
            'height': '5.2',
            'onpix_total': '1.8',
            'x_bar': '8.0',
            'y_bar': '13.0',
            'x2_bar': '0.5',
            'y2_bar': '6.0',
            'xy_bar': '6.0',
            'x2y_bar': '10.0',
            'xy2_bar': '8.0',
            'x_ege': '0.0',
            'xegvy': '8.0',
            'y_ege': '0.0',
            'yegvx': '8.0'
        }
        
        encoded_packet = processor.encode_single_packet(sample_packet)
        print(f"  Encoded packet shape: {encoded_packet.shape}")
        print(f"  Encoded packet values: {encoded_packet[0][:5]}...")
        
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nSummary:")
        print(f"  ✓ Generated 2 datasets with 10,000 rows each")
        print(f"  ✓ Successfully loaded and processed large dataset")
        print(f"  ✓ Data types handled correctly (categorical + numeric)")
        print(f"  ✓ Preprocessing and encoding working properly")
        print(f"\nFiles created:")
        print(f"  1. {natural_path}")
        print(f"  2. {balanced_path}")
        print(f"\nTotal test time: {(datetime.now() - start_time).total_seconds():.2f} seconds")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def generate_custom_sized_dataset(n_samples=10000, save_path="data/custom_dataset.csv"):
    """Generate a dataset with custom number of samples"""
    
    print(f"\nGenerating custom dataset with {n_samples} samples...")
    
    np.random.seed(42)
    
    # Letters
    letters = ['T', 'I', 'D', 'A', 'B', 'C', 'E', 'F', 'G', 'H']
    
    # Generate data
    data = {
        'Letter': np.random.choice(letters, n_samples),
        'x_box': np.abs(np.random.randn(n_samples) * 3 + 4),
        'y_box': np.abs(np.random.randn(n_samples) * 4 + 8),
        'width': np.abs(np.random.randn(n_samples) * 2 + 3),
        'height': np.abs(np.random.randn(n_samples) * 2.5 + 5),
        'onpix_total': np.abs(np.random.randn(n_samples) * 3 + 4),
        'x_bar': np.abs(np.random.randn(n_samples) * 3 + 7),
        'y_bar': np.abs(np.random.randn(n_samples) * 3 + 8),
        'x2_bar': np.abs(np.random.randn(n_samples) * 2 + 4),
        'y2_bar': np.abs(np.random.randn(n_samples) * 2.5 + 6),
        'xy_bar': np.abs(np.random.randn(n_samples) * 3 + 7),
        'x2y_bar': np.abs(np.random.randn(n_samples) * 2.5 + 6),
        'xy2_bar': np.abs(np.random.randn(n_samples) * 2.5 + 7),
        'x_ege': np.abs(np.random.randn(n_samples) * 1.5 + 2),
        'xegvy': np.abs(np.random.randn(n_samples) * 2 + 6),
        'y_ege': np.abs(np.random.randn(n_samples) * 1.5 + 2),
        'yegvx': np.abs(np.random.randn(n_samples) * 2 + 7),
    }
    
    # Create labels based on feature patterns
    labels = []
    for i in range(n_samples):
        # Use feature combinations to determine label
        if data['x_bar'][i] > 9 or data['y_bar'][i] > 10:
            labels.append('attack')
        elif data['width'][i] > 5 or data['height'][i] > 7:
            labels.append('attack')
        elif data['onpix_total'][i] > 6:
            labels.append('attack')
        else:
            labels.append('normal')
    
    data['Label'] = labels
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"  ✓ Custom dataset saved to: {save_path}")
    
    return df

if __name__ == "__main__":
    # Run the main test
    success = test_data_loading_with_large_dataset()
    
    # Optionally generate additional custom datasets
    print("\n" + "="*80)
    print("GENERATING ADDITIONAL DATASETS")
    print("="*80)
    
    # Generate datasets of different sizes
    sizes = [5000, 20000]
    for size in sizes:
        generate_custom_sized_dataset(size, f"data/custom_dataset_{size}.csv")
    
    print("\n" + "="*80)
    print("ALL DATASETS GENERATED SUCCESSFULLY!")
    print("="*80)
    print("\nDataset files created:")
    print("  • data/natural_distribution_10000.csv (10,000 rows - natural distribution)")
    print("  • data/balanced_distribution_10000.csv (10,000 rows - balanced classes)")
    print("  • data/custom_dataset_5000.csv (5,000 rows)")
    print("  • data/custom_dataset_20000.csv (20,000 rows)")
    print("\nYou can now use these datasets with the Intrusion Detection System!")