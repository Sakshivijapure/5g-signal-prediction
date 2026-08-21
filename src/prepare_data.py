import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/fishnet_30x30_processed_feature_engineered.csv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

FEATURES = [
    "NDVI_center",
    "Building_Coverage",
    "Weighted_Height",
    "ALT_M_",
    "Base_Central_frequency_point",
    "Base_Electronic_downtilt",
    "Base_Mechanical_downtilt",
    "Base_Power",
    "Match_Dist",
    "DEM_center",
    "True_3D_Dist",
    "Match_Angle_sin",
    "Match_Angle_cos",
    "Base_Direction_angle_sin",
    "Base_Direction_angle_cos"
]

TARGET = "SS_RSRP"

df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates()

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

train_df = X_train.copy()
train_df[TARGET] = y_train

test_df = X_test.copy()
test_df[TARGET] = y_test

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print("Training data:", train_df.shape)
print("Testing data:", test_df.shape)