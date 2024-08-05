# importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb

# data preprocessing
df = pd.read_csv("housing.csv")
scaler_pt = PowerTransformer()
part_2 = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "median_house_value"]
df[part_2] = scaler_pt.fit_transform(df[part_2])

# creating data for training and testing
X, y = df[["longitude", "latitude", "housing_median_age", "total_rooms", "population", "households", "median_income"]], df["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# training model
model = xgb.XGBRegressor(max_depth=7, device="cuda")
model.fit(X_train, y_train)

# testing model
y_pred = model.predict(X_test)
r2 = r2_score(y_pred, y_test)

print(r2) # quality
