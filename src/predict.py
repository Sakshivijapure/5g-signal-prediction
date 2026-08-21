import pandas as pd
import joblib


MODEL_PATH = "models/best_model.pkl"


model = joblib.load(MODEL_PATH)


data = {
    "NDVI_center": 0.3,
    "Building_Coverage": 0.5,
    "Weighted_Height": 20,
    "ALT_M_": 100,
    "Base_Central_frequency_point": 350000,
    "Base_Electronic_downtilt": 6,
    "Base_Mechanical_downtilt": 2,
    "Base_Power": 20,
    "Match_Dist": 500,
    "DEM_center": 100,
    "True_3D_Dist": 510,
    "Match_Angle_sin": 0.5,
    "Match_Angle_cos": 0.866,
    "Base_Direction_angle_sin": 0.5,
    "Base_Direction_angle_cos": 0.866
}


input_data = pd.DataFrame([data])

prediction = model.predict(input_data)

print("Predicted SS-RSRP:", prediction[0], "dBm")