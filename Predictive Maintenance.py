#Predictive maintenance
# Project 6: Predict the number of remaining operational cycles before failure for Turbofan engine
'''Data set Link:
https://drive.google.com/file/d/1dgWM0KKOnoN9kVObbA-GahsgXPJBCT4c/view?usp=sharing'''

# STEP 1: Install & Import
   #!pip install pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# STEP 2: Extract ZIP
zip_path = "/content/drive/MyDrive/data/Project6_Pdm_Predict the number of remaining operational cycles before failure for Turbofan engine.zip"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/content/drive/MyDrive/data")

print("Files extracted:", os.listdir("/content/drive/MyDrive/data"))

# STEP 3: Define Columns
cols = ['unit', 'cycle'] + [f'op{i}' for i in range(1,4)] + [f's{i}' for i in range(1,22)]

# STEP 4: Load Data
train = pd.read_csv('/content/drive/MyDrive/data/train_FD001.txt', sep=' ', header=None)
train = train.dropna(axis=1)
train.columns = cols

test = pd.read_csv('/content/drive/MyDrive/data/test_FD001.txt', sep=' ', header=None)
test = test.dropna(axis=1)
test.columns = cols

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# STEP 5: Create RUL (Target)
rul = train.groupby('unit')['cycle'].max().reset_index()
rul.columns = ['unit', 'max_cycle']

train = train.merge(rul, on='unit')
train['RUL'] = train['max_cycle'] - train['cycle']
train.drop(['max_cycle'], axis=1, inplace=True)

# STEP 6: Normalize Data
scaler = MinMaxScaler()
features = train.columns.drop(['unit', 'cycle', 'RUL'])

train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])

# STEP 7: Train Model
X = train[features]
y = train['RUL']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# STEP 8: Evaluate Model
train_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, train_pred))

print("\nModel Evaluation:")
print("RMSE:", rmse)

# STEP 9: Predict on Test Data
pred = model.predict(test[features])

print("\nSample Predictions (RUL):")
print(pred[:10])

# STEP 10: Visualization
plt.figure()
plt.plot(y.values[:100], label='Actual RUL')
plt.plot(train_pred[:100], label='Predicted RUL')
plt.legend()
plt.title("Actual vs Predicted RUL (First 100)")
plt.xlabel("Samples")
plt.ylabel("RUL")
plt.show()

# Project 7: Predict life time of a bearing in manufacturing industry
'''Data set Link:
https://drive.google.com/file/d/12rV9AhpqbMivYCu4WVM7DhXpO1aO98k_/view?usp=sharing'''

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# Feature Extraction Function
def extract_features(signal):
    features = []

    features.append(np.mean(signal))
    features.append(np.std(signal))
    features.append(np.max(signal))
    features.append(np.min(signal))
    features.append(np.ptp(signal))
    features.append(np.sqrt(np.mean(signal**2)))  # RMS
    features.append(np.mean(np.abs(signal)))
    features.append(np.mean(signal**2))

    return features


# Load Dataset
def load_data(data_path):
    files = sorted(os.listdir(data_path))
    X = []

    print(f"Processing {len(files)} files...")

    for file in tqdm(files):
        file_path = os.path.join(data_path, file)

        try:
            data = np.loadtxt(file_path)
            signal = data[:, 0]  # Using first channel

            features = extract_features(signal)
            X.append(features)

        except:
            continue

    return np.array(X)



# Create RUL Labels

def create_rul_labels(num_samples):
    return np.arange(num_samples, 0, -1)



# Main Pipeline for 1st_test
def main():
    data_path = "/content/drive/MyDrive/data2/1st_test"   # change if needed

    X = load_data(data_path)
    print("Feature Shape:", X.shape)

    y = create_rul_labels(len(X))

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    print("MAE:", mae)

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(y_test, label="Actual RUL")
    plt.plot(y_pred, label="Predicted RUL")
    plt.legend()
    plt.title("RUL Prediction")
    plt.show()

if __name__ == '__main__':
  main()

# Main Pipeline for 2nd_test
def main():
    data_path = "/content/drive/MyDrive/data2/2nd_test"   # change if needed

    X = load_data(data_path)
    print("Feature Shape:", X.shape)

    y = create_rul_labels(len(X))

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    print("MAE:", mae)

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(y_test, label="Actual RUL")
    plt.plot(y_pred, label="Predicted RUL")
    plt.legend()
    plt.title("RUL Prediction")
    plt.show()

if __name__ == '__main__':
  main()

#Project 8: Predictive maintenance of Gearbox using vibration sensors

'''Data set Link:
https://drive.google.com/file/d/1nNNnjMPntlo5X0t_cif7cmlJhikCyWyP/view?usp=sharing'''

# STEP 1: Import Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# STEP 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# STEP 3: Dataset Path
data_path = "/content/drive/MyDrive/data3/gearbox_dataset"

print("Folders inside dataset:", os.listdir(data_path))

# STEP 4: Feature Extraction
def extract_features(signal):
    return [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        np.median(signal),
        np.percentile(signal, 25),
        np.percentile(signal, 75),
        np.sqrt(np.mean(signal**2))  # RMS
    ]

# STEP 5: Load Dataset

X = []
y = []

conditions = ["Healthy Data", "BrokenToothData"]

for label, condition in enumerate(conditions):
    folder = os.path.join(data_path, condition)
    print(f"\nProcessing: {condition}")

    for file in tqdm(os.listdir(folder)):

        if file.endswith(".txt"):
            file_path = os.path.join(folder, file)

            try:
                # ✅ Read raw lines
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                data = []

                for line in lines:
                    # split by space/tab
                    values = line.strip().split()

                    # convert to float if possible
                    row = []
                    for v in values:
                        try:
                            row.append(float(v))
                        except:
                            pass

                    if len(row) > 0:
                        data.append(row)

                if len(data) == 0:
                    print("Skipped (no numeric data):", file)
                    continue

                df = pd.DataFrame(data)

                # Extract features
                for col in df.columns:
                    signal = df[col].values

                    if len(signal) == 0:
                        continue

                    features = extract_features(signal)

                    X.append(features)
                    y.append(label)

            except Exception as e:
                print("Error in file:", file, e)

# Convert to numpy
X = np.array(X)
y = np.array(y)

print("\nDataset shape:", X.shape)


# STEP 6: Check if data loaded
if len(X) == 0:
    raise ValueError("❌ No data loaded. Check file format!")

# STEP 7: Normalize
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# STEP 8: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 9: Train Model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# STEP 10: Prediction
y_pred = model.predict(X_test)

# STEP 11: Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# STEP 12: Feature Importance
plt.figure()
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.title("Feature Importance")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()
