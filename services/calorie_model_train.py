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
    # Load new dataset with Vietnamese nutrition columns
    df = pd.read_csv("data/food_nutrition_data_final.csv")
    
    # Map Vietnamese column names to standard names used in model
    X = pd.DataFrame({
        "Protein_g": df["Protein"],
        "Carbohydrates_g": df["Glucid"],  # Glucid = carbohydrates
        "Fat_g": df["Lipid"],             # Lipid = fat
        "Fiber_g": df["Celluloza"],       # Celluloza = fiber
        "Sugars_g": df["Glucid"] * 0.1    # Estimate sugars as 10% of total carbs if not available
    }).fillna(0)
    
    y = df["Năng lượng"]  # Energy/calories column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    rmse = math.sqrt(mse)
    print(f"✅ RMSE: {rmse:.2f}")

    out = Path("services/models/calorie_from_macro_rf.pkl")
    out.parent.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
    joblib.dump(model, out)
    print(f"✅ Model saved to {out}")

if __name__ == "__main__":
    main()
