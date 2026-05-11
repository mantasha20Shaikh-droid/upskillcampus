# Industrial Manufacturing and Production
# Project 10: Quality Prediction in a Mining Process
''' Data set Link:
 https://drive.google.com/file/d/1N80d8eTDAf1JMQXGQbHDAUaMGRyA8QG3/view?usp=sharing'''

# STEP 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# STEP 2: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# STEP 3: File Paths
data10_path = "/content/drive/MyDrive/Colab Notebooks/data5/MiningProcess_Flotation_Plant_Database.csv"
data11_path = "/content/drive/MyDrive/Colab Notebooks/data6/continuous_factory_process.csv"

# STEP 4: Safe Loader
def load_data(path, name):
    print(f"\nLoading {name}...")
    df = pd.read_csv(path, engine='python', on_bad_lines='skip')
    print(f"{name} Shape:", df.shape)
    return df

df10 = load_data(data10_path, "Mining Dataset")
df11 = load_data(data11_path, "Manufacturing Dataset")

# PROJECT 10: Mining
print("\n===== PROJECT 10: Mining =====")

df10.columns = df10.columns.str.strip()

# Handle datetime
if 'date' in df10.columns:
    df10['date'] = pd.to_datetime(df10['date'], errors='coerce')
    df10 = df10.sort_values('date')

# Drop weak rows
df10 = df10.dropna(thresh=int(0.7 * df10.shape[1]))

# Target
target10 = df10.columns[-1]

X10 = df10.drop(columns=[target10])
y10 = df10[target10]

# Convert comma decimals → numeric
X10 = X10.astype(str).replace(',', '.', regex=True)
X10 = X10.apply(pd.to_numeric, errors='coerce')
X10 = X10.dropna(axis=1, how='all')
X10 = X10.fillna(X10.mean())

y10 = y10.astype(str).str.replace(',', '.')
y10 = pd.to_numeric(y10, errors='coerce')
y10 = y10.fillna(y10.mean())

# Split
X_train10, X_test10, y_train10, y_test10 = train_test_split(
    X10, y10, test_size=0.2, random_state=42
)

# Scale
scaler10 = StandardScaler()
X_train10 = scaler10.fit_transform(X_train10)
X_test10 = scaler10.transform(X_test10)

# Model
model10 = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
model10.fit(X_train10, y_train10)

# Predict
y_pred10 = model10.predict(X_test10)

# Evaluation
print("\n--- Mining Results ---")
print("MAE:", mean_absolute_error(y_test10, y_pred10))
print("R2 Score:", r2_score(y_test10, y_pred10))
print("RMSE:", np.sqrt(mean_squared_error(y_test10, y_pred10)))

# Scatter Plot
plt.figure()
plt.scatter(y_test10, y_pred10)
plt.xlabel("Actual Silica")
plt.ylabel("Predicted Silica")
plt.title("Mining Prediction")
plt.show()

# Trend Plot (Bonus)
plt.figure()
plt.plot(y_test10.values[:100], label="Actual")
plt.plot(y_pred10[:100], label="Predicted")
plt.legend()
plt.title("Mining Trend Comparison")
plt.show()

# Project 11: Multi-stage continuous-flow manufacturing process
''' Data set Link:
 https://drive.google.com/file/d/1yvZzslpbWw2mpCVF5QqueSkNrNHmtvDE/view?usp=share_link'''

print("\n===== PROJECT 11: Manufacturing =====")

df11.columns = df11.columns.str.strip()

df11 = df11.dropna(thresh=int(0.7 * df11.shape[1]))

target11 = df11.columns[-1]

X11 = df11.drop(columns=[target11])
y11 = df11[target11]

# Convert comma decimals → numeric
X11 = X11.astype(str).replace(',', '.', regex=True)
X11 = X11.apply(pd.to_numeric, errors='coerce')
X11 = X11.dropna(axis=1, how='all')
X11 = X11.fillna(X11.mean())

y11 = y11.astype(str).str.replace(',', '.')
y11 = pd.to_numeric(y11, errors='coerce')
y11 = y11.fillna(y11.mean())

# Split
X_train11, X_test11, y_train11, y_test11 = train_test_split(
    X11, y11, test_size=0.2, random_state=42
)

# Scale
scaler11 = StandardScaler()
X_train11 = scaler11.fit_transform(X_train11)
X_test11 = scaler11.transform(X_test11)

# Model
model11 = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
model11.fit(X_train11, y_train11)

# Predict
y_pred11 = model11.predict(X_test11)

# Evaluation
print("\n--- Manufacturing Results ---")
print("MAE:", mean_absolute_error(y_test11, y_pred11))
print("R2 Score:", r2_score(y_test11, y_pred11))
print("RMSE:", np.sqrt(mean_squared_error(y_test11, y_pred11)))

# Scatter Plot
plt.figure()
plt.scatter(y_test11, y_pred11)
plt.xlabel("Actual Output")
plt.ylabel("Predicted Output")
plt.title("Manufacturing Prediction")
plt.show()

# FEATURE IMPORTANCE
print("\n===== FEATURE IMPORTANCE (Mining) =====")

importances = model10.feature_importances_
features = X10.columns

plt.figure(figsize=(4,6))
plt.barh(features, importances)
plt.title("Feature Importance")
plt.show()

# FINAL MESSAGE
print("\n SUCCESS: Both Projects Completed Without Errors!")
