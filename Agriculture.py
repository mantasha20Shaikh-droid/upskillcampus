# A. Agriculture
# Project 4 : Prediction of Agriculture Crop Production in India.

# ------------------- Import Libraries -------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# ------------------- Create Dataset -------------------
data = [
    ["ARHAR","Uttar Pradesh",9794.05,23076.74,1941.55,9.83,"Total Foodgrains",
     158.8,168.6,171.3,159.4,178.9,
     128.5,128.8,127.6,126.0,131.7,
     123.6,130.9,134.3,126.5,135.9,
     "Paddy","Chinsurah Rice (IET 19140)","Medium",
     "Zone1","Agricultural Production","Annual","Ton",
     *([np.nan]*12),
     198.36,208.60,217.28,230.77,234.46,
     218.10,244.49,259.28,257.13,264.38
    ],

    ["ARHAR","Karnataka",10593.15,16528.68,2172.46,7.47,"Rice",
     200.8,207.9,213.3,191.6,206.4,
     168.5,168.9,175.1,161.2,164.8,
     119.2,123.1,121.8,118.9,125.2,
     "Paddy","CNI 383-5-11","Low",
     "Zone2","Agricultural Production","Annual","Ton",
     *([np.nan]*12),
     103.30,109.87,110.57,120.95,118.13,
     103.95,120.85,131.27,128.07,129.36
    ]
]

columns = [
    "Crop","State","Cost_A2_FL","Cost_C2","Cost_Production_C2","Yield","Category",
    "Production_2006_07","Production_2007_08","Production_2008_09","Production_2009_10","Production_2010_11",
    "Area_2006_07","Area_2007_08","Area_2008_09","Area_2009_10","Area_2010_11",
    "Yield_2006_07","Yield_2007_08","Yield_2008_09","Yield_2009_10","Yield_2010_11",
    "Crop_2","Variety","Season","Recommended_Zone",
    "Particulars","Frequency","Unit",
    "1993","1994","1995","1996","1997","1998",
    "1999","2000","2001","2002","2003","2004",
    "2005","2006","2007","2008","2009","2010",
    "2011","2012","2013","2014"
]

df = pd.DataFrame(data, columns=columns)

# ------------------- EDA -------------------
print("Dataset Preview:\n", df.head())
print("\nInfo:\n")
print(df.info())

# ------------------- Data Cleaning -------------------

# Drop columns with all NaN (1993–2004)
df.drop(columns=[str(year) for year in range(1993, 2005)], inplace=True)

# Drop unnecessary text-heavy columns
df.drop(["Crop_2", "Variety", "Recommended_Zone", "Particulars"], axis=1, inplace=True)

# Fill remaining missing values
df.fillna(0, inplace=True)

# ------------------- Encoding -------------------
categorical_cols = ['Crop','State','Category','Season','Frequency','Unit']

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ------------------- Feature Selection -------------------
y = df["Production_2010_11"]

X = df.drop(["Production_2010_11"], axis=1)

# Optional: Reduce features (recommended for tiny dataset)
X = X[[
    "Cost_A2_FL",
    "Cost_C2",
    "Cost_Production_C2",
    "Yield",
    "Production_2006_07",
    "Production_2007_08",
    "Production_2008_09",
    "Production_2009_10"
]]

# ------------------- Model Training -------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ------------------- Prediction -------------------
y_pred = model.predict(X)

# ------------------- Output -------------------
print("\nActual Values:", y.values)
print("Predicted Values:", y_pred)

# ------------------- Feature Importance -------------------
importance = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importance:\n", importance.sort_values(ascending=False))

# ------------------- Visualization -------------------

# 1. Cost Comparison
plt.figure(figsize=(8,5))
df[['Cost_A2_FL','Cost_C2','Cost_Production_C2']].plot(kind='bar')
plt.title("Cost Comparison")
plt.xlabel("Index")
plt.ylabel("Cost")
plt.xticks(rotation=0)
plt.legend()
plt.show()


# 2. Yield Comparison by State
plt.figure(figsize=(6,4))
sns.barplot(x='State', y='Yield', data=df)
plt.title("Yield by State")
plt.show()


# 3. Production Trend (2006–2010)
years = ['Production_2006_07','Production_2007_08','Production_2008_09',
         'Production_2009_10','Production_2010_11']

plt.figure(figsize=(8,5))
for i in range(len(df)):
    plt.plot(years, df.loc[i, years], marker='o', label=f"Sample {i}")

plt.title("Production Trend")
plt.xlabel("Year")
plt.ylabel("Production")
plt.legend()
plt.xticks(rotation=45)
plt.show()


# 4. Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# 5. Feature Importance (after model training)
importance = pd.Series(model.feature_importances_, index=X.columns)

plt.figure(figsize=(8,5))
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()
     

# A. Agriculture
# Project 5 : Crop and Weed Detection.

# Install YOLO
   #!pip install ultralytics

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create data.yaml file
with open('data.yaml', 'w') as f:
    f.write("""
path: /content/drive/MyDrive/Project5_Ag_Crop_and_weed
train: images/train
val: images/val

names:
  0: crop
  1: weed
""")

# Check dataset
import os

train_path = "/content/drive/MyDrive/Project5_Ag_Crop_and_weed/images/train"

if os.path.exists(train_path):
    print("✅ Dataset found")
    print("Total training images:", len(os.listdir(train_path)))
else:
    print(" Dataset NOT found. Check folder name.")

# Train YOLO model
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='data.yaml',
    epochs=50,
    imgsz=512
)

# Prediction
results = model.predict(
    source='/content/drive/MyDrive/Project5_Ag_Crop_and_weed/images/val',
    save=True
)

# Show output
from IPython.display import Image, display
import glob

output_images = glob.glob('runs/detect/predict/*.jpg')

if output_images:
    display(Image(filename=output_images[0]))
else:
    print(" No output images found")

     
