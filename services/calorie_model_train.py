"""
Train RandomForestRegressor to predict Calories from macros.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math
import joblib
from pathlib import Path

def main():
    df = pd.read_csv("data/cleaned_logs.csv")
    X = df[["Protein_g", "Carbohydrates_g", "Fat_g", "Fiber_g", "Sugars_g"]].fillna(0)
    y = df["Calories_kcal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    rmse = math.sqrt(mse)
    print(f"✅ RMSE: {rmse:.2f}")

    out = Path("services/models/calorie_from_macro_rf.pkl")
    joblib.dump(model, out)
    print(f"✅ Model saved to {out}")

if __name__ == "__main__":
    main()
