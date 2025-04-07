# c:/price/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import joblib
from datetime import datetime
import glob
import os
import pyarrow.parquet as pq
from tqdm import tqdm
import time

def process_chunk(chunk):
    """Process a single chunk of data"""
    # Create a copy of the chunk to avoid SettingWithCopyWarning
    chunk = chunk.copy()
    
    # Convert trip_time from seconds to minutes
    chunk.loc[:, 'trip_duration'] = chunk['trip_time'] / 60
    
    # Filter for valid trips
    chunk = chunk[
        (chunk['trip_miles'] > 0) & 
        (chunk['trip_miles'] < 100) &  # Reasonable distance limit
        (chunk['trip_duration'] > 0) & 
        (chunk['trip_duration'] < 180) &  # 3 hours max
        (chunk['base_passenger_fare'] > 0) & 
        (chunk['base_passenger_fare'] < 500)  # Reasonable fare limit
    ]
    
    # Convert pickup datetime
    chunk.loc[:, 'pickup_datetime'] = pd.to_datetime(chunk['pickup_datetime'])
    
    # Extract time features
    chunk.loc[:, 'hour'] = chunk['pickup_datetime'].dt.hour
    chunk.loc[:, 'day_of_week'] = chunk['pickup_datetime'].dt.dayofweek
    chunk.loc[:, 'month'] = chunk['pickup_datetime'].dt.month
    
    # Create time-based features
    chunk.loc[:, 'is_peak_hour'] = chunk['hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    chunk.loc[:, 'is_weekend'] = (chunk['day_of_week'] >= 5).astype(int)
    chunk.loc[:, 'is_night'] = ((chunk['hour'] >= 22) | (chunk['hour'] <= 5)).astype(int)
    
    # Create season groups
    chunk.loc[:, 'season'] = pd.cut(chunk['month'], 
                           bins=[0, 3, 6, 9, 12],
                           labels=['winter', 'spring', 'summer', 'fall'])
    
    # Calculate speed
    chunk.loc[:, 'speed'] = chunk['trip_miles'] / (chunk['trip_duration'] / 60)
    
    # Create company-specific features
    chunk.loc[:, 'is_uber'] = (chunk['hvfhs_license_num'] == 'HV0003').astype(int)
    chunk.loc[:, 'is_lyft'] = (chunk['hvfhs_license_num'] == 'HV0005').astype(int)
    
    # Calculate base price components
    chunk.loc[:, 'base_price'] = chunk['trip_miles'] * 2.5 + chunk['trip_duration'] * 0.5
    
    # Create company-specific prices
    chunk.loc[:, 'uber_base_price'] = chunk['base_price'] * chunk['is_uber']
    chunk.loc[:, 'lyft_base_price'] = chunk['base_price'] * chunk['is_lyft']
    
    # Add surge pricing features
    chunk.loc[:, 'uber_peak_multiplier'] = chunk['is_peak_hour'] * 1.5 * chunk['is_uber']
    chunk.loc[:, 'lyft_peak_multiplier'] = chunk['is_peak_hour'] * 1.3 * chunk['is_lyft']
    chunk.loc[:, 'uber_weekend_multiplier'] = chunk['is_weekend'] * 1.3 * chunk['is_uber']
    chunk.loc[:, 'lyft_weekend_multiplier'] = chunk['is_weekend'] * 1.2 * chunk['is_lyft']
    
    # Add minimum fare features
    chunk.loc[:, 'uber_min_fare'] = 7.0 * chunk['is_uber']
    chunk.loc[:, 'lyft_min_fare'] = 8.0 * chunk['is_lyft']
    
    # Add airport-specific features
    chunk.loc[:, 'is_airport_pickup'] = chunk['PULocationID'].isin([132, 138]).astype(int)
    chunk.loc[:, 'is_airport_dropoff'] = chunk['DOLocationID'].isin([132, 138]).astype(int)
    
    # Add airport surcharges
    chunk.loc[:, 'airport_surcharge'] = (chunk['is_airport_pickup'] | chunk['is_airport_dropoff']) * 5.0
    
    # Create interaction terms
    chunk.loc[:, 'miles_duration'] = chunk['trip_miles'] * chunk['trip_duration']
    chunk.loc[:, 'miles_speed'] = chunk['trip_miles'] * chunk['speed']
    chunk.loc[:, 'duration_speed'] = chunk['trip_duration'] * chunk['speed']
    
    # Company-specific interaction terms
    chunk.loc[:, 'uber_miles_duration'] = chunk['miles_duration'] * chunk['is_uber']
    chunk.loc[:, 'lyft_miles_duration'] = chunk['miles_duration'] * chunk['is_lyft']
    
    return chunk

def prepare_features(chunk):
    """Prepare features for a chunk of data"""
    # Define numerical features
    numerical_features = [
        'trip_miles', 'trip_duration', 'speed', 'base_price',
        'uber_base_price', 'lyft_base_price', 'uber_peak_multiplier',
        'lyft_peak_multiplier', 'uber_weekend_multiplier',
        'lyft_weekend_multiplier', 'airport_surcharge',
        'miles_duration', 'miles_speed', 'duration_speed',
        'uber_miles_duration', 'lyft_miles_duration'
    ]
    
    # Define categorical features
    categorical_features = [
        'hour', 'day_of_week', 'month', 'season',
        'is_peak_hour', 'is_weekend', 'is_night',
        'is_uber', 'is_lyft', 'is_airport_pickup',
        'is_airport_dropoff'
    ]
    
    # Create feature matrix
    X = pd.concat([
        chunk[numerical_features],
        pd.get_dummies(chunk[categorical_features], drop_first=True)
    ], axis=1)
    
    return X, numerical_features, categorical_features

def train_model():
    """Train the fare prediction model using cross-validation and incremental learning"""
    print("Loading and preprocessing data in chunks...")
    start_time = time.time()
    
    # Load all parquet files
    parquet_files = glob.glob("data/raw/fhvhv_tripdata_*.parquet")
    print(f"Found {len(parquet_files)} parquet files to process")
    
    # Initialize model with warm_start
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        warm_start=True
    )
    
    # Initialize scaler
    scaler = StandardScaler()
    numerical_features = None
    categorical_features = None
    
    # Initialize lists to store cross-validation scores
    cv_scores = []
    total_rows = 0
    chunk_count = 0
    
    # Process each file in chunks
    for file_idx, file in enumerate(parquet_files, 1):
        print(f"\nProcessing file {file_idx}/{len(parquet_files)}: {os.path.basename(file)}")
        file_start_time = time.time()
        
        try:
            parquet_file = pq.ParquetFile(file)
            num_row_groups = parquet_file.num_row_groups
            print(f"File has {num_row_groups} row groups")
            
            # Process each batch in the file
            for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=50000), 1):
                print(f"Processing batch {batch_idx}...", end='\r')
                chunk_start_time = time.time()
                
                chunk = batch.to_pandas()
                
                # Process the chunk
                chunk = process_chunk(chunk)
                total_rows += len(chunk)
                chunk_count += 1
                
                # Prepare features for the chunk
                X_chunk, num_features, cat_features = prepare_features(chunk)
                
                # Store features if not already stored
                if numerical_features is None:
                    numerical_features = num_features
                    categorical_features = cat_features
                
                # Scale numerical features
                X_chunk[numerical_features] = scaler.fit_transform(X_chunk[numerical_features])
                
                # Perform cross-validation on this chunk
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                chunk_scores = cross_val_score(
                    model,
                    X_chunk,
                    chunk['base_passenger_fare'],
                    cv=kf,
                    scoring='r2',
                    n_jobs=-1
                )
                cv_scores.extend(chunk_scores)
                
                # Train on this chunk
                model.fit(X_chunk, chunk['base_passenger_fare'])
                
                # Clear memory
                del chunk
                del X_chunk
                
                # Print progress and timing
                chunk_time = time.time() - chunk_start_time
                if chunk_count % 5 == 0:  # Print every 5 chunks
                    print(f"\nProcessed {chunk_count} chunks, {total_rows:,} rows")
                    print(f"Average time per chunk: {chunk_time:.2f} seconds")
                    print(f"Current mean R² score: {np.mean(cv_scores):.4f}")
            
            file_time = time.time() - file_start_time
            print(f"\nCompleted file {file_idx}/{len(parquet_files)} in {file_time:.2f} seconds")
            
        except Exception as e:
            print(f"\nError processing file {file}: {str(e)}")
            continue
    
    # Calculate and print final metrics
    total_time = time.time() - start_time
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    
    print(f"\nTraining completed in {total_time:.2f} seconds")
    print(f"Total rows processed: {total_rows:,}")
    print(f"Total chunks processed: {chunk_count}")
    print(f"Average time per chunk: {total_time/chunk_count:.2f} seconds")
    print(f"\nCross-validation results:")
    print(f"Mean R² score: {mean_cv_score:.4f} (+/- {std_cv_score * 2:.4f})")
    
    # Save model artifacts
    model_artifacts = {
        'model': model,
        'scaler': scaler,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'cv_scores': cv_scores,
        'mean_cv_score': mean_cv_score,
        'std_cv_score': std_cv_score,
        'total_rows': total_rows,
        'total_time': total_time
    }
    
    # Create models directory if it doesn't exist
    os.makedirs("data/models", exist_ok=True)
    
    # Save the model artifacts
    joblib.dump(model_artifacts, "data/models/gradient_boosting_fare_model.joblib")
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()