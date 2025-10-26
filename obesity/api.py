from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, confloat, constr
from typing import Literal
import pandas as pd
import uvicorn
from obesity.test_model import predict_obesity

app = FastAPI(
    title="Obesity Prediction API",
    description="API để dự đoán mức độ béo phì dựa trên các thông số của người dùng",
    version="1.0.0"
)

class ObesityInput(BaseModel):
    Gender: Literal['Male', 'Female'] = Field(..., description="Giới tính")
    Age: float = Field(..., gt=0, description="Tuổi")
    Height: float = Field(..., gt=0, lt=3, description="Chiều cao (mét)")
    Weight: float = Field(..., gt=0, description="Cân nặng (kg)")
    family_history_with_overweight: Literal['yes', 'no'] = Field(..., description="Tiền sử gia đình có người thừa cân")
    FAVC: Literal['yes', 'no'] = Field(..., description="Thường xuyên ăn thức ăn có calo cao")
    FCVC: float = Field(..., ge=1, le=3, description="Tần suất ăn rau (1-3)")
    NCP: float = Field(..., ge=1, le=4, description="Số bữa ăn chính trong ngày (1-4)")
    CAEC: Literal['Always', 'Sometimes', 'Frequently'] = Field(..., description="Thói quen ăn vặt giữa các bữa")
    SMOKE: Literal['yes', 'no'] = Field(..., description="Hút thuốc")
    CH2O: float = Field(..., ge=1, le=3, description="Lượng nước uống hàng ngày (lít)")
    SCC: Literal['yes', 'no'] = Field(..., description="Theo dõi lượng calo tiêu thụ")
    FAF: float = Field(..., ge=0, le=3, description="Tần suất hoạt động thể chất (0-3)")
    TUE: float = Field(..., ge=0, le=2, description="Thời gian sử dụng thiết bị điện tử (giờ/ngày)")
    CALC: Literal['Always', 'Sometimes', 'no'] = Field(..., description="Tiêu thụ đồ uống có cồn")
    MTRANS: Literal['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'] = Field(..., description="Phương tiện di chuyển thường dùng")

    class Config:
        schema_extra = {
            "example": {
                "Gender": "Female",
                "Age": 25.0,
                "Height": 1.65,
                "Weight": 70.0,
                "family_history_with_overweight": "yes",
                "FAVC": "no",
                "FCVC": 2.0,
                "NCP": 3.0,
                "CAEC": "Sometimes",
                "SMOKE": "no",
                "CH2O": 2.5,
                "SCC": "no",
                "FAF": 1.0,
                "TUE": 1.0,
                "CALC": "no",
                "MTRANS": "Public_Transportation"
            }
        }

@app.post("/predict", 
         response_model=dict,
         description="Dự đoán mức độ béo phì dựa trên các thông số đầu vào",
         response_description="Kết quả dự đoán mức độ béo phì")
async def predict(data: ObesityInput):
    try:
        # Chuyển input thành DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # Dự đoán
        prediction = predict_obesity(input_df)
        
        # Tính BMI
        bmi = data.Weight / (data.Height ** 2)
        
        return {
            "prediction": prediction[0],
            "bmi": round(bmi, 2),
            "bmi_category": get_bmi_category(bmi)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Thiếu cân"
    elif bmi < 24.9:
        return "Bình thường"
    elif bmi < 29.9:
        return "Thừa cân"
    elif bmi < 34.9:
        return "Béo phì độ I"
    elif bmi < 39.9:
        return "Béo phì độ II"
    else:
        return "Béo phì độ III"

@app.get("/", response_model=dict)
async def root():
    return {
        "message": "Welcome to Obesity Prediction API",
        "docs": "/docs for Swagger UI",
        "redoc": "/redoc for ReDoc"
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)