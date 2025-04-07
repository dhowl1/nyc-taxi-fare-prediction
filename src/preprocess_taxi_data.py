# c:/price/py.py
import pandas as pd
import numpy as np
import glob
import os
import pyarrow.parquet as pq
from sklearn.preprocessing import RobustScaler

# Create directories if they don't exist
os.makedirs("data/temp", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Define season groups
season_map = {1: 'winter', 2: 'winter', 12: 'winter',
              3: 'spring', 4: 'spring', 5: 'spring', 6: 'spring',
              7: 'summer', 8: 'summer', 9: 'summer',
              10: 'fall', 11: 'fall'}

# Load enriched taxi zone lookup
df_zones = pd.read_csv("data/raw/taxi_zone_lookup_enriched.csv")
required_columns = ['LocationID', 'Latitude', 'Longitude']
if not all(col in df_zones.columns for col in required_columns):
    raise ValueError("taxi_zone_lookup_enriched.csv must contain 'LocationID', 'Latitude', and 'Longitude' columns.")

# Step 1: Process each file individually and save to temp files
files = glob.glob("data/raw/fhvhv_tripdata_2024-*.parquet")
sample_per_file = 5000  # Increased sample size
total_files = len(files)

total_rows = sample_per_file * total_files
final_sample_size = min(50000, total_rows)  # Increased final sample size

print(f"Sampling {sample_per_file} rows per file, {total_rows} total rows, final sample: {final_sample_size} rows")

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    R = 3959.87433  # Earth's radius in miles
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def is_peak_hour(hour):
    return 1 if hour in [7, 8, 9, 16, 17, 18, 19] else 0

def is_weekend(day):
    return 1 if day >= 5 else 0

def is_airport_location(location_id):
    return 1 if location_id in [132, 138] else 0  # JFK and LGA

def safe_division(x, y, default=0.0):
    try:
        result = x / y if y != 0 else default
        return result if np.isfinite(result) else default
    except:
        return default

for file in files:
    print(f"Processing: {file}")
    filename = os.path.basename(file)
    month = int(filename.split('-')[1].split('.')[0])
    
    parquet_file = pq.ParquetFile(file)
    file_samples = []
    rows_collected = 0
    
    for batch in parquet_file.iter_batches(batch_size=100000):
        chunk = batch.to_pandas()
        
        # Basic filtering
        chunk['company'] = chunk['hvfhs_license_num'].map({'HV0003': 'Uber', 'HV0005': 'Lyft'})
        chunk = chunk[chunk['company'].isin(['Uber', 'Lyft'])]
        
        # Remove extreme outliers in base_passenger_fare
        fare_q1 = chunk['base_passenger_fare'].quantile(0.01)
        fare_q3 = chunk['base_passenger_fare'].quantile(0.99)
        fare_iqr = fare_q3 - fare_q1
        chunk = chunk[
            (chunk['base_passenger_fare'] >= fare_q1 - 1.5 * fare_iqr) &
            (chunk['base_passenger_fare'] <= fare_q3 + 1.5 * fare_iqr) &
            (chunk['base_passenger_fare'] > 0)  # Ensure positive fares
        ]
        
        # Remove trips with unrealistic values
        chunk = chunk[chunk['trip_miles'] > 0]  # Remove 0-mile trips
        chunk['pickup_datetime'] = pd.to_datetime(chunk['pickup_datetime'])
        chunk['dropoff_datetime'] = pd.to_datetime(chunk['dropoff_datetime'])
        chunk['trip_duration_hours'] = (chunk['dropoff_datetime'] - chunk['pickup_datetime']).dt.total_seconds() / 3600
        chunk = chunk[
            (chunk['trip_duration_hours'] > 0) &  # Remove 0-duration trips
            (chunk['trip_duration_hours'] <= 24)  # Remove trips longer than 24 hours
        ]
        
        # Calculate speed with bounds
        chunk['speed_mph'] = chunk.apply(
            lambda row: safe_division(row['trip_miles'], row['trip_duration_hours'], 0.0),
            axis=1
        )
        chunk = chunk[chunk['speed_mph'] <= 100]  # Remove unrealistic speeds
        
        if len(chunk) > 0:
            sample_size = min(sample_per_file - rows_collected, len(chunk))
            if sample_size > 0:
                file_samples.append(chunk.sample(sample_size, random_state=42))
                rows_collected += sample_size
        if rows_collected >= sample_per_file:
            break
    
    if not file_samples:
        print(f"No samples collected for {file}, skipping...")
        continue
    
    df_file = pd.concat(file_samples, ignore_index=True)
    
    # Enhanced feature engineering
    df_file['trip_duration'] = (df_file['dropoff_datetime'] - df_file['pickup_datetime']).dt.total_seconds() / 60
    df_file['pickup_hour'] = df_file['pickup_datetime'].dt.hour
    df_file['day_of_week'] = df_file['pickup_datetime'].dt.dayofweek
    df_file['is_peak_hour'] = df_file['pickup_hour'].apply(is_peak_hour)
    df_file['is_weekend'] = df_file['day_of_week'].apply(is_weekend)
    df_file['season'] = season_map[month]
    
    # Airport-specific features
    df_file['is_airport_pickup'] = df_file['PULocationID'].apply(is_airport_location)
    df_file['is_airport_dropoff'] = df_file['DOLocationID'].apply(is_airport_location)
    df_file['is_airport_trip'] = (df_file['is_airport_pickup'] | df_file['is_airport_dropoff']).astype(int)
    
    # Merge with taxi zones to get coordinates
    df_file = df_file.merge(df_zones[['LocationID', 'Latitude', 'Longitude']], 
                           left_on='PULocationID', right_on='LocationID', how='left')
    df_file = df_file.merge(df_zones[['LocationID', 'Latitude', 'Longitude']], 
                           left_on='DOLocationID', right_on='LocationID', how='left',
                           suffixes=('_pu', '_do'))
    
    # Calculate direct distance with handling for missing coordinates
    df_file['direct_distance'] = df_file.apply(
        lambda row: calculate_haversine_distance(
            row['Latitude_pu'], row['Longitude_pu'],
            row['Latitude_do'], row['Longitude_do']
        ) if all(pd.notna([row['Latitude_pu'], row['Longitude_pu'], 
                          row['Latitude_do'], row['Longitude_do']])) else 0.0,
        axis=1
    )
    
    # Calculate derived features with safe division
    df_file['route_efficiency'] = df_file.apply(
        lambda row: safe_division(row['trip_miles'], row['direct_distance'], 1.0),
        axis=1
    )
    df_file['speed'] = df_file.apply(
        lambda row: safe_division(row['trip_miles'], row['trip_duration'] / 60, 0.0),
        axis=1
    )
    
    # Calculate fare components
    df_file['fare_per_mile'] = df_file.apply(
        lambda row: safe_division(row['base_passenger_fare'], row['trip_miles'], 0.0),
        axis=1
    )
    df_file['fare_per_minute'] = df_file.apply(
        lambda row: safe_division(row['base_passenger_fare'], row['trip_duration'], 0.0),
        axis=1
    )
    
    # Calculate total fare including all fees
    df_file['total_fare'] = (
        df_file['base_passenger_fare'] +
        df_file['tolls'] +
        df_file['bcf'] +
        df_file['sales_tax'] +
        df_file['congestion_surcharge'] +
        df_file['airport_fee']
    )
    
    # Calculate minimum fares
    df_file['min_fare'] = df_file.apply(
        lambda row: 7.0 if row['company'] == 'Uber' else 8.0,
        axis=1
    )
    
    # Clip derived features to reasonable ranges
    df_file['route_efficiency'] = df_file['route_efficiency'].clip(0.5, 5.0)
    df_file['speed'] = df_file['speed'].clip(0, 100)
    df_file['fare_per_mile'] = df_file['fare_per_mile'].clip(0, 50)
    df_file['fare_per_minute'] = df_file['fare_per_minute'].clip(0, 10)
    
    # Convert categorical variables
    df_file['company'] = df_file['company'].astype('category')
    df_file['season'] = df_file['season'].astype('category')
    
    # Save to temporary file
    temp_file = f"data/temp/processed_2024-{month:02d}.csv"
    df_file.to_csv(temp_file, index=False)
    print(f"Saved: {temp_file}")
    
    del df_file
    del file_samples

# Step 2: Merge all temporary files into one
temp_files = glob.glob("data/temp/processed_2024-*.csv")
df_list = []
for temp_file in temp_files:
    df_temp = pd.read_csv(temp_file)
    df_list.append(df_temp)

df_combined = pd.concat(df_list, ignore_index=True)

# Final sample with stratification by fare ranges
df_combined['fare_range'] = pd.qcut(df_combined['total_fare'], q=10, labels=False)
df_sample = df_combined.groupby('fare_range', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), final_sample_size//10), random_state=42)
)

# Save the final combined file
df_sample.to_csv("data/processed/combined_data_sample.csv", index=False)
print("Final combined file saved: data/processed/combined_data_sample.csv")

# Clean up temporary files
for temp_file in temp_files:
    os.remove(temp_file)
print("Temporary files cleaned up.")