# C. Smart city
# Project 9: Forecasting of Smart city traffic patterns
# We are working with the government to transform various cities into a smart city. The vision is to convert it into a digital and intelligent city to improve the efficiency of services for the citizens. One of the problems faced by the government is traffic. You are a data scientist working to manage the traffic of the city better and to provide input on infrastructure planning for the future.
# The government wants to implement a robust traffic system for the city by being prepared for traffic peaks. They want to understand the traffic patterns of the four junctions of the city. Traffic patterns on holidays, as well as on various other occasions during the year, differ from normal working days. This is important to take into account for your forecasting.

# Data set Link:
# https://drive.google.com/file/d/1y61cDyuO9Zrp1fSchWcAmCxk0B6SMx7X/view?usp=sharing

# STEP 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# STEP 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# STEP 3: Load TRAIN & TEST Data
train_path = '/content/drive/MyDrive/data4/train_aWnotuB.csv'
test_path  = '/content/drive/MyDrive/data4/test_BdBKkAj.csv'

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

print("Train Data Loaded")
print(train_df.head())

print("\nTest Data Loaded")
print(test_df.head())

# STEP 4: Preprocessing
    # Convert DateTime
train_df['DateTime'] = pd.to_datetime(train_df['DateTime'])
test_df['DateTime']  = pd.to_datetime(test_df['DateTime'])

    # Sort
train_df = train_df.sort_values('DateTime')
test_df  = test_df.sort_values('DateTime')

    # Feature Engineering
for df in [train_df, test_df]:
    df['year'] = df['DateTime'].dt.year
    df['month'] = df['DateTime'].dt.month
    df['day'] = df['DateTime'].dt.day
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek

    # Fill missing
train_df.fillna(method='ffill', inplace=True)
test_df.fillna(method='ffill', inplace=True)

# STEP 5: Visualization (Train Data)
plt.figure(figsize=(12,5))
sns.lineplot(data=train_df, x='DateTime', y='Vehicles', hue='Junction')
plt.title("Traffic Pattern (Train Data)")
plt.show()

# STEP 6: Model for Each Junction
predictions = []

for junc in train_df['Junction'].unique():

    print(f"\nProcessing Junction {junc}")

    # Filter data
    train_j = train_df[train_df['Junction'] == junc].copy()
    test_j  = test_df[test_df['Junction'] == junc].copy()

    # Set index
    train_j.set_index('DateTime', inplace=True)

    # Time series
    ts = train_j['Vehicles'].resample('H').mean()
    ts.fillna(method='ffill', inplace=True)

    # Build ARIMA model
    model = ARIMA(ts, order=(5,1,0))
    model_fit = model.fit()

    # Forecast for test length
    forecast = model_fit.forecast(steps=len(test_j))

    # Save predictions
    test_j['Vehicles_Predicted'] = forecast.values
    predictions.append(test_j)
# STEP 7: Combine All Predictions
final_df = pd.concat(predictions)

print("\nFinal Predictions:")
print(final_df.head())

# STEP 8: Save Output
output_path = '/content/drive/MyDrive/data4/traffic_predictions.csv'
final_df.to_csv(output_path, index=False)

print("\n Predictions saved to Drive:", output_path)

# STEP 9: Visualization (Example Junction)
j1 = final_df[final_df['Junction'] == 1]

plt.figure(figsize=(10,5))
plt.plot(j1['DateTime'], j1['Vehicles_Predicted'], label='Predicted')
plt.legend()
plt.title("Predicted Traffic - Junction 1")
plt.show()

# DONE
print("\n Smart City Traffic Forecasting Completed!")
     
