"""
Setup script to train all models and prepare the application.
Run this script after cloning the repository to set up all required models.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_models():
    """Train all models and prepare the application."""
    print("[INFO] Setting up Nutrition AI Application...")
    print("=" * 50)
    
    # Create necessary directories
    print("[INFO] Creating directories...")
    directories = [
        "models/obesity", 
        "models/meal_plan",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created directory: {directory}")
    
    # Check if data file exists
    data_file = project_root / "data" / "food_nutrition_data_final.csv"
    if not data_file.exists():
        print(f"[ERROR] Data file not found at {data_file}")
        print("Please ensure you have the nutrition dataset in the data/ directory")
        return False
    
    print(f"[OK] Found data file: {data_file}")
    
    # Train obesity model
    print("\n[INFO] Training obesity prediction model...")
    try:
        from scripts.train_obesity_model import train_obesity_model
        train_obesity_model()
        print("[OK] Obesity model training completed")
    except Exception as e:
        print(f"[ERROR] Error training obesity model: {e}")
        return False
    
    # Train meal plan model
    print("\n[INFO] Training meal plan recommendation model...")
    try:
        from scripts.train_meal_plan_model import train_meal_plan_model
        train_meal_plan_model()
        print("[OK] Meal plan model training completed")
    except Exception as e:
        print(f"[ERROR] Error training meal plan model: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("[SUCCESS] Setup completed successfully!")
    print("You can now run the application with: python run_server.py")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = setup_models()
    if not success:
        sys.exit(1)
