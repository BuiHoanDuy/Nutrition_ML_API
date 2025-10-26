from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd

from services.obesity_inference import predict_obesity

router = APIRouter(
    prefix="/obesity",
    tags=["obesity"],
    responses={404: {"description": "Not found"}},
)

# Định nghĩa model input
class ObesityInput(BaseModel):
    Gender: Literal['Male', 'Female'] = Field(..., title="Giới tính", description="Male = Nam, Female = Nữ")
    Age: float = Field(..., gt=0, title="Tuổi", description="Tuổi (số, đơn vị: năm)")
    Height: float = Field(..., gt=0, lt=3, title="Chiều cao", description="Chiều cao (mét)")
    Weight: float = Field(..., gt=0, title="Cân nặng", description="Cân nặng (kg)")
    family_history_with_overweight: Literal['yes', 'no'] = Field(..., title="Tiền sử gia đình", description="Gia đình có người thừa cân (yes/no)")
    FAVC: Literal['yes', 'no'] = Field(..., title="Ăn đồ nhiều calo", description="Ăn đồ ăn nhiều calo/fast food thường xuyên? (yes/no)")
    FCVC: float = Field(..., ge=1, le=3, title="Ăn rau củ", description="Tần suất ăn rau (1-3)")
    NCP: float = Field(..., ge=1, le=4, title="Số bữa/ngày", description="Số bữa chính trong ngày (1-4)")
    CAEC: Literal['Always', 'Sometimes', 'Frequently'] = Field(..., title="Ăn giữa bữa", description="Thói quen ăn vặt giữa các bữa")
    SMOKE: Literal['yes', 'no'] = Field(..., title="Hút thuốc", description="Có hút thuốc không (yes/no)")
    CH2O: float = Field(..., ge=1, le=3, title="Lượng nước/ngày", description="Lượng nước uống hàng ngày (lít)")
    SCC: Literal['yes', 'no'] = Field(..., title="Theo dõi calo", description="Có theo dõi lượng calo tiêu thụ không (yes/no)")
    FAF: float = Field(..., ge=0, le=3, title="Tần suất tập thể dục", description="Tần suất hoạt động thể chất (0-3)")
    TUE: float = Field(..., ge=0, le=2, title="Thời gian thiết bị", description="Thời gian sử dụng thiết bị điện tử (giờ/ngày)")
    CALC: Literal['Always', 'Sometimes', 'no'] = Field(..., title="Uống rượu", description="Tiêu thụ đồ uống có cồn (Always/Sometimes/no)")
    MTRANS: Literal['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'] = Field(..., title="Phương tiện di chuyển", description="Phương tiện di chuyển thường dùng")

    model_config = {
        "json_schema_extra": {
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
        }}

# Bản đồ tên trường -> nhãn tiếng Việt và mô tả ngắn
FIELD_LABELS = {
    "Gender": {
        "label": "Giới tính",
        "description": "Male = Nam, Female = Nữ",
        "example": "Female"
    },
    "Age": {"label": "Tuổi", "description": "Tuổi (số, đơn vị: năm)", "example": 25},
    "Height": {"label": "Chiều cao", "description": "Chiều cao tính bằng mét (m)", "example": 1.65},
    "Weight": {"label": "Cân nặng", "description": "Cân nặng tính bằng kilogam (kg)", "example": 70},
    "family_history_with_overweight": {"label": "Tiền sử gia đình", "description": "Gia đình có người thừa cân (yes/no)", "example": "yes"},
    "FAVC": {"label": "Ăn đồ ăn nhiều calo/fast food", "description": "Thường xuyên ăn đồ ăn nhanh? (yes/no)", "example": "no"},
    "FCVC": {"label": "Ăn rau củ", "description": "Tần suất ăn rau (1-3)", "example": 2},
    "NCP": {"label": "Số bữa/ngày", "description": "Số bữa chính trong ngày (1-4)", "example": 3},
    "CAEC": {"label": "Ăn giữa bữa", "description": "Thói quen ăn vặt giữa các bữa (Always/Sometimes/Frequently)", "example": "Sometimes"},
    "SMOKE": {"label": "Hút thuốc", "description": "Có hút thuốc không (yes/no)", "example": "no"},
    "CH2O": {"label": "Lượng nước/ngày", "description": "Lượng nước uống hàng ngày (lít, 1-3)", "example": 2.5},
    "SCC": {"label": "Theo dõi calo", "description": "Có theo dõi lượng calo tiêu thụ không (yes/no)", "example": "no"},
    "FAF": {"label": "Tần suất tập thể dục", "description": "Tần suất hoạt động thể chất (0-3)", "example": 1},
    "TUE": {"label": "Thời gian thiết bị", "description": "Thời gian sử dụng thiết bị điện tử (giờ/ngày, 0-2)", "example": 1},
    "CALC": {"label": "Uống rượu", "description": "Tiêu thụ đồ uống có cồn (Always/Sometimes/no)", "example": "no"},
    "MTRANS": {"label": "Phương tiện di chuyển", "description": "Public_Transportation/Walking/Automobile/Motorbike/Bike", "example": "Public_Transportation"}
}

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

@router.post("/predict", 
         response_model=dict,
         description="Dự đoán mức độ béo phì dựa trên các thông số đầu vào",
         response_description="Kết quả dự đoán mức độ béo phì")
async def predict_obesity_endpoint(data: ObesityInput):
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

@router.get("/fields", 
         response_model=dict,
         description="Danh sách các trường đầu vào (tiếng Việt)")
async def get_fields():
    """Trả về bản đồ tên trường sang nhãn tiếng Việt và ví dụ mẫu để dễ hình dung."""
    return FIELD_LABELS