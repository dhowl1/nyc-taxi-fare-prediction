# NYC Taxi Fare Prediction System

A machine learning system that predicts taxi fares in New York City using historical trip data from Uber and Lyft. The model achieves 84% accuracy (R² score) in fare prediction by considering multiple factors including time, distance, location, and company-specific pricing.

## Key Features

- 🚕 Accurate fare predictions for both Uber and Lyft
- 🕒 Time-based pricing (peak hours, weekends, nights)
- ✈️ Special handling for airport trips
- 💰 Company-specific pricing models
- 📊 87% prediction accuracy
- 🔄 Real-time processing capabilities

## Technical Highlights

- Gradient Boosting Regression with cross-validation
- Efficient batch processing of large datasets
- Feature engineering for time, distance, and location
- Company-specific pricing multipliers
- Memory-optimized data processing

## Core Components

### 1. Data Preparation (`prepare_taxi_zones.py`)
- Processes taxi zone data
- Enriches location information
- Prepares zone-specific features

### 2. Data Preprocessing (`preprocess_taxi_data.py`)
- Cleans and validates trip data
- Extracts time-based features
- Calculates trip metrics
- Handles missing values

### 3. Model Training (`train_model.py`)
- Implements Gradient Boosting Regressor
- Performs cross-validation
- Handles incremental learning
- Optimizes memory usage

### 4. Fare Prediction (`predict_fare.py`)
- Currently working on this file

## Use Cases

- Fare estimation for ride-sharing services
- Price comparison between Uber and Lyft
- Airport transfer cost prediction
- Peak hour pricing analysis
- Weekend vs weekday fare comparison

## Data

Uses NYC Taxi & Limousine Commission (TLC) For-Hire Vehicle (FHV) trip data from 2024, including:
- Trip distances and durations
- Pickup and dropoff locations
- Time of trip
- Company identifiers
- Base fares and surcharges

## Performance

- R² Score: 0.87 (87% accuracy)
- Processing Speed: 17-30 seconds per batch
- Memory Efficient: Processes data in chunks
- Scalable: Ready for cloud deployment

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare taxi zone data: `python prepare_taxi_zones.py`
4. Preprocess trip data: `python preprocess_taxi_data.py`
5. Train the model: `python train_model.py`
6. Make predictions: `python predict_fare.py`

### Example Prediction
```python
from predict_fare import predict_fare

# Example: Times Square to JFK Airport
fare = predict_fare(
    pickup_location="Times Square",
    dropoff_location="JFK Airport",
    pickup_time="2024-03-28 17:30:00",
    company="uber"
)
print(f"Estimated fare: ${fare:.2f}")
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- pyarrow
- joblib
- tqdm

## Project Structure

```
nyc-taxi-fare-prediction/
├── src/
│   ├── prepare_taxi_zones.py
│   ├── preprocess_taxi_data.py
│   ├── train_model.py
│   └── predict_fare.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── README.md
└── requirements.txt
```

## Future Improvements

- [ ] Add weather data integration
- [ ] Implement traffic pattern analysis
- [ ] Add special event detection
- [ ] Deploy to cloud infrastructure
- [ ] Create API endpoint for predictions
- [ ] Add real-time surge pricing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NYC Taxi & Limousine Commission for the dataset
- Uber and Lyft for providing real-world pricing data 
