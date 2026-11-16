import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

data = pd.read_csv("../datasets/mushrooms.csv")
target = "class"

print(data.shape)
print(data.head())
print(data.isnull().sum())

data = data.replace("?", np.nan)
data = data.drop_duplicates()

for col in data.columns:
  if data[col].isnull().sum() > 0:
    if pd.api.types.is_numeric_dtype(data[col]):
      data[col] = data[col].fillna(data[col].median())
    else:
      data[col] = data[col].fillna(data[col].mode()[0])

y = LabelEncoder().fit_transform(data[target])
X = data.drop(columns=[target])
X = pd.get_dummies(X, drop_first=True).astype(float)
names = list(X.columns)
X = X.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rows, cols = X_train.shape

print(rows, cols)
print(np.bincount(y))
