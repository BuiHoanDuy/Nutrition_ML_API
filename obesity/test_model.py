import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Đường dẫn đến thư mục chứa model đã train
MODEL_DIR = "obesity/output"

def load_artifacts():
    """Load các artifacts đã lưu từ quá trình training"""
    with open(os.path.join(MODEL_DIR, "final_model.pkl"), "rb") as f:
        artifacts = pickle.load(f)
    return (artifacts["model"], 
            artifacts["label_encoders"], 
            artifacts["scaler"],
            {i: label for i, label in enumerate(artifacts["target_le"].classes_)})

def preprocess_input(data, label_encoders, scaler):
    """Tiền xử lý dữ liệu đầu vào giống như lúc training"""
    # Fill NA nếu có
    for c in data.columns:
        if data[c].dtype in [np.float64, np.int64]:
            data[c] = data[c].fillna(data[c].median())
        else:
            data[c] = data[c].fillna(data[c].mode().iloc[0])
    
    # Chuyển đổi categorical features
    cat_cols = data.select_dtypes(include=['object','bool']).columns.tolist()
    for c in cat_cols:
        if c in label_encoders:
            data[c] = label_encoders[c].transform(data[c].astype(str))
    
    # Tính BMI
    if 'Height' in data.columns and 'Weight' in data.columns:
        data['BMI_computed'] = data['Weight'] / (data['Height'] ** 2)
    
    # Scale numeric features
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if scaler is not None:
        data[num_cols] = scaler.transform(data[num_cols])
    
    return data

def predict_obesity(data):
    """Dự đoán phân loại béo phì cho dữ liệu đầu vào"""
    # Load artifacts
    model, label_encoders, scaler, target_mapping = load_artifacts()
    
    # Tiền xử lý dữ liệu
    processed_data = preprocess_input(data.copy(), label_encoders, scaler)
    
    # Dự đoán
    predictions = model.predict(processed_data)
    
    # Chuyển đổi predictions về nhãn gốc
    predictions = [target_mapping[pred] for pred in predictions]
    
    return predictions

if __name__ == "__main__":
    # Ví dụ sử dụng
    test_data = pd.DataFrame({
        'Gender': ['Female'],
        'Age': [25.0],
        'Height': [1.65],
        'Weight': [70.0],
        'family_history_with_overweight': ['yes'],
        'FAVC': ['no'],
        'FCVC': [2.0],
        'NCP': [3.0],
        'CAEC': ['Sometimes'],
        'SMOKE': ['no'],
        'CH2O': [2.5],
        'SCC': ['no'],
        'FAF': [1.0],
        'TUE': [1.0],
        'CALC': ['no'],
        'MTRANS': ['Public_Transportation']
    })
    
    predictions = predict_obesity(test_data)
    print("\nDữ liệu test:")
    print(test_data)
    print("\nKết quả dự đoán:", predictions[0])

    # Test thêm một vài trường hợp khác
    more_test_data = pd.DataFrame({
        'Gender': ['Male', 'Female'],
        'Age': [30.0, 22.0],
        'Height': [1.80, 1.60],
        'Weight': [90.0, 55.0],
        'family_history_with_overweight': ['yes', 'no'],
        'FAVC': ['yes', 'no'],
        'FCVC': [3.0, 2.0],
        'NCP': [4.0, 3.0],
        'CAEC': ['Sometimes', 'Frequently'],
        'SMOKE': ['no', 'no'],
        'CH2O': [2.0, 1.5],
        'SCC': ['no', 'no'],
        'FAF': [0.0, 2.0],
        'TUE': [1.0, 0.0],
        'CALC': ['Sometimes', 'no'],
        'MTRANS': ['Public_Transportation', 'Walking']
    })
    
    predictions = predict_obesity(more_test_data)
    print("\nThêm dữ liệu test:")
    print(more_test_data)
    print("\nKết quả dự đoán:")
    for i, pred in enumerate(predictions):
        print(f"Người {i+1}: {pred}")