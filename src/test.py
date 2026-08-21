import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/random_forest.pkl"

TARGET = "SS_RSRP"


df = pd.read_csv(TEST_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]


model = joblib.load(MODEL_PATH)

predictions = model.predict(X)


mae = mean_absolute_error(y, predictions)
rmse = mean_squared_error(y, predictions) ** 0.5
r2 = r2_score(y, predictions)


print("Model evaluation completed!")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)