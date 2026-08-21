import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_validate, KFold


TRAIN_PATH = "data/processed/train.csv"
MODEL_PATH = "models/best_model.pkl"
TARGET = "SS_RSRP"


df = pd.read_csv(TRAIN_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]


models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}


cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = []

for name, model in models.items():

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2"
        }
    )

    results.append({
        "Model": name,
        "MAE": -scores["test_mae"].mean(),
        "RMSE": -scores["test_rmse"].mean(),
        "R2": scores["test_r2"].mean()
    })


results_df = pd.DataFrame(results)

print("\nModel comparison:")
print(results_df.to_string(index=False))


best_model_name = results_df.loc[
    results_df["RMSE"].idxmin(),
    "Model"
]

best_model = models[best_model_name]

best_model.fit(X, y)

joblib.dump(best_model, MODEL_PATH)

print("\nBest model:", best_model_name)
print("Model saved to:", MODEL_PATH)