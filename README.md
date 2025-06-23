# Day-1-internship-task-1

 # 🚗 Automobile Dataset - Data Cleaning & Machine Learning Prep

This repository demonstrates how to clean and prepare a real-world automobile dataset for machine learning. It includes code to handle missing values, convert data types, encode categorical features, normalize data, detect outliers, and perform train-test split.

---

## 📁 Dataset
File: `Automobile_data.csv`

---

## 🔧 Steps Covered

### ✅ 1. Load Dataset and Replace '?' with NaN
```python
import pandas as pd
import numpy as np

df = pd.read_csv('Automobile_data.csv')
df.replace('?', np.nan, inplace=True)
```

### ✅ 2. Convert Columns to Numeric
```python
numeric_columns = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm', 'price']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
```

### ✅ 3. Handle Missing Values
```python
df['normalized-losses'].fillna(df['normalized-losses'].mean(), inplace=True)
df['bore'].fillna(df['bore'].mean(), inplace=True)
df['stroke'].fillna(df['stroke'].mean(), inplace=True)
df['horsepower'].fillna(df['horsepower'].mean(), inplace=True)
df['peak-rpm'].fillna(df['peak-rpm'].mean(), inplace=True)
df['price'].fillna(df['price'].median(), inplace=True)
df['num-of-doors'].fillna(df['num-of-doors'].mode()[0], inplace=True)
```

### ✅ 4. Encode Categorical Variables
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['fuel-type'] = le.fit_transform(df['fuel-type'])
df['aspiration'] = le.fit_transform(df['aspiration'])
df['num-of-doors'] = le.fit_transform(df['num-of-doors'])

df = pd.get_dummies(df, columns=[
    'make', 'body-style', 'drive-wheels',
    'engine-location', 'engine-type', 'fuel-system'
], drop_first=True)

df['num-of-cylinders'] = df['num-of-cylinders'].replace({
    'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12
})
```

### ✅ 5. Normalize Numeric Columns
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numeric_to_scale = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm', 'engine-size', 'price']
df[numeric_to_scale] = scaler.fit_transform(df[numeric_to_scale])
```

### ✅ 6. Visualize and Remove Outliers
```python
import seaborn as sns
import matplotlib.pyplot as plt

for col in numeric_to_scale:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
```

### ✅ 7. Train-Test Split
```python
from sklearn.model_selection import train_test_split

X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 📚 Output
Cleaned dataset ready for machine learning with all preprocessing steps applied.

---


