# TIỀN XỬ LÝ VÀ CHUẨN HÓA DỮ LIỆU OBESITYDATASET.CSV

## 1. TỔNG QUAN

Bộ dữ liệu `ObesityDataSet.csv` được sử dụng để xây dựng mô hình dự đoán mức độ béo phì dựa trên các đặc điểm cá nhân và thói quen sinh hoạt. File này mô tả chi tiết quy trình tiền xử lý và chuẩn hóa dữ liệu từ dữ liệu thô sang dữ liệu sẵn sàng cho việc huấn luyện mô hình.

### Thông tin cơ bản:
- **File gốc**: `data/ObesityDataSet.csv`
- **File đã tiền xử lý**: `data/obesity_preprocessed.csv`
- **Số mẫu**: 2,111 mẫu (sau khi loại bỏ dữ liệu không hợp lệ: 2,087 mẫu)
- **Số cột gốc**: 17 cột
- **Số cột sau tiền xử lý**: 34 cột

---

## 2. CẤU TRÚC DỮ LIỆU GỐC

### 2.1. Các cột trong dataset gốc:

#### **Cột số (Numeric):**
1. **Age** (float64): Tuổi (14-61 tuổi)
2. **Height** (float64): Chiều cao (mét, 1.45-1.98m)
3. **Weight** (float64): Cân nặng (kg)
4. **FCVC** (float64): Tần suất ăn rau củ (1-3)
5. **NCP** (float64): Số bữa ăn chính trong ngày (1-4)
6. **CH2O** (float64): Lượng nước uống hàng ngày (lít, 1-3)
7. **FAF** (float64): Tần suất hoạt động thể chất (0-3)
8. **TUE** (float64): Thời gian sử dụng thiết bị điện tử (0-2)

#### **Cột phân loại (Categorical):**
9. **Gender** (object): Giới tính - `['Female', 'Male']`
10. **family_history_with_overweight** (object): Tiền sử gia đình thừa cân - `['yes', 'no']`
11. **FAVC** (object): Ăn đồ nhiều calo/fast food - `['no', 'yes']`
12. **CAEC** (object): Thói quen ăn vặt giữa các bữa - `['Sometimes', 'Frequently', 'Always', 'no']`
13. **SMOKE** (object): Hút thuốc - `['no', 'yes']`
14. **SCC** (object): Theo dõi lượng calo tiêu thụ - `['no', 'yes']`
15. **CALC** (object): Uống đồ uống có cồn - `['no', 'Sometimes', 'Frequently', 'Always']`
16. **MTRANS** (object): Phương tiện di chuyển - `['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike']`

#### **Cột mục tiêu (Target):**
17. **NObeyesdad** (object): Mức độ béo phì - `['Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Insufficient_Weight', 'Obesity_Type_II', 'Obesity_Type_III']`

### 2.2. Thống kê mô tả dữ liệu gốc:

```
Age:       Mean=24.31, Std=6.35, Min=14, Max=61
Height:    Mean=1.70,  Std=0.09,  Min=1.45, Max=1.98
Weight:    Mean=86.59, Std=26.19, Min=39, Max=173
FCVC:      Mean=2.42,  Std=0.53,  Min=1, Max=3
NCP:       Mean=2.69,  Std=0.87,  Min=1, Max=4
CH2O:      Mean=2.01,  Std=0.61,  Min=1, Max=3
FAF:       Mean=1.01,  Std=0.85,  Min=0, Max=3
TUE:       Mean=0.66,  Std=0.61,  Min=0, Max=2
```

### 2.3. Kiểm tra dữ liệu:
- ✅ **Không có giá trị thiếu (Missing values)**: Tất cả 17 cột đều không có giá trị null
- ✅ **Không có giá trị ngoại lai nghiêm trọng**: Các giá trị đều nằm trong phạm vi hợp lý
- ⚠️ **Cần loại bỏ một số mẫu**: 24 mẫu bị loại bỏ do dữ liệu không hợp lệ (có thể do BMI quá cao/thấp hoặc giá trị không nhất quán)

---

## 3. QUY TRÌNH TIỀN XỬ LÝ

### 3.1. Bước 1: Tính toán đặc trưng mới (Feature Engineering)

#### **Tính BMI (Body Mass Index):**
```python
BMI = Weight / (Height ** 2)
```

- **Mục đích**: BMI là chỉ số quan trọng để đánh giá mức độ béo phì
- **Đơn vị**: kg/m²
- **Phạm vi**: Thường từ 15-50 kg/m² trong dataset này

**Cột mới được thêm**: `BMI`

---

### 3.2. Bước 2: Chuẩn hóa dữ liệu số (Standardization)

Tất cả các cột số được chuẩn hóa bằng phương pháp **Z-score normalization** (Standardization):

```python
normalized_value = (value - mean) / std
```

#### **Các cột được chuẩn hóa:**
1. `Age`
2. `Height`
3. `Weight`
4. `FCVC`
5. `NCP`
6. `CH2O`
7. `FAF`
8. `TUE`
9. `BMI` (cột mới tính toán)

#### **Lý do chuẩn hóa:**
- ✅ Đưa tất cả các đặc trưng về cùng một thang đo (mean=0, std=1)
- ✅ Tránh các đặc trưng có giá trị lớn chi phối mô hình
- ✅ Cải thiện hiệu suất của các thuật toán machine learning (đặc biệt là Random Forest, Neural Networks)
- ✅ Giúp gradient descent hội tụ nhanh hơn (nếu sử dụng)

#### **Ví dụ chuẩn hóa:**
```
Age gốc: 21 tuổi
Mean(Age) = 24.31
Std(Age) = 6.35
Age chuẩn hóa = (21 - 24.31) / 6.35 = -0.5266
```

**Kết quả**: Tất cả giá trị số sau chuẩn hóa có phân phối chuẩn với mean ≈ 0 và std ≈ 1.

---

### 3.3. Bước 3: Mã hóa dữ liệu phân loại (Categorical Encoding)

Tất cả các cột phân loại được chuyển đổi sang **One-Hot Encoding** (dummy variables).

#### **Các cột phân loại được mã hóa:**

1. **Gender** → `Gender_Female`, `Gender_Male`
2. **family_history_with_overweight** → `family_history_with_overweight_no`, `family_history_with_overweight_yes`
3. **FAVC** → `FAVC_no`, `FAVC_yes`
4. **CAEC** → `CAEC_Always`, `CAEC_Frequently`, `CAEC_Sometimes`, `CAEC_no`
5. **SMOKE** → `SMOKE_no`, `SMOKE_yes`
6. **SCC** → `SCC_no`, `SCC_yes`
7. **CALC** → `CALC_Always`, `CALC_Frequently`, `CALC_Sometimes`, `CALC_no`
8. **MTRANS** → `MTRANS_Automobile`, `MTRANS_Bike`, `MTRANS_Motorbike`, `MTRANS_Public_Transportation`, `MTRANS_Walking`

#### **Cách hoạt động của One-Hot Encoding:**

Mỗi giá trị phân loại được chuyển thành một vector nhị phân (binary vector) với:
- `1` cho giá trị đúng của mẫu đó
- `0` cho tất cả các giá trị khác

**Ví dụ:**
```
Gender = "Female"
  → Gender_Female = 1 (True)
  → Gender_Male = 0 (False)

CAEC = "Sometimes"
  → CAEC_Always = 0 (False)
  → CAEC_Frequently = 0 (False)
  → CAEC_Sometimes = 1 (True)
  → CAEC_no = 0 (False)
```

#### **Lý do sử dụng One-Hot Encoding:**
- ✅ Không tạo ra thứ tự giả giữa các giá trị phân loại (khác với Label Encoding)
- ✅ Phù hợp với các thuật toán ML không xử lý tốt dữ liệu phân loại trực tiếp
- ✅ Mỗi giá trị được biểu diễn độc lập, không có mối quan hệ thứ bậc

---

### 3.4. Bước 4: Xử lý cột mục tiêu (Target Variable)

Cột `NObeyesdad` được giữ nguyên trong dataset đã tiền xử lý để làm nhãn cho việc huấn luyện.

**Các giá trị có thể:**
- `Normal_Weight`
- `Overweight_Level_I`
- `Overweight_Level_II`
- `Obesity_Type_I`
- `Insufficient_Weight`
- `Obesity_Type_II`
- `Obesity_Type_III`

**Lưu ý**: Trong quá trình huấn luyện, cột này được mã hóa bằng `LabelEncoder` để chuyển thành số nguyên (0-6) cho mô hình.

---

## 4. CẤU TRÚC DỮ LIỆU SAU TIỀN XỬ LÝ

### 4.1. Tổng số cột: 34 cột

#### **Nhóm 1: Cột số đã chuẩn hóa (9 cột)**
```
Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE, BMI
```

#### **Nhóm 2: Cột phân loại đã One-Hot (23 cột)**
```
Gender_Female, Gender_Male
family_history_with_overweight_no, family_history_with_overweight_yes
FAVC_no, FAVC_yes
CAEC_Always, CAEC_Frequently, CAEC_Sometimes, CAEC_no
SMOKE_no, SMOKE_yes
SCC_no, SCC_yes
CALC_Always, CALC_Frequently, CALC_Sometimes, CALC_no
MTRANS_Automobile, MTRANS_Bike, MTRANS_Motorbike, 
MTRANS_Public_Transportation, MTRANS_Walking
```

#### **Nhóm 3: Cột mục tiêu (2 cột)**
```
NObeyesdad (giữ nguyên giá trị gốc)
NObeyesdad_Label (mã hóa số, nếu có)
```

### 4.2. Ví dụ dữ liệu sau tiền xử lý:

**Dòng đầu tiên:**
```
Age: -0.5266
Height: -0.8874
Weight: -0.8730
...
Gender_Female: True
Gender_Male: False
family_history_with_overweight_yes: True
...
NObeyesdad: "Normal_Weight"
```

---

## 5. QUY TRÌNH CHUẨN HÓA TRONG INFERENCE

Khi sử dụng mô hình để dự đoán, dữ liệu đầu vào cũng phải được chuẩn hóa theo cùng cách:

### 5.1. Chuẩn hóa số:

```python
# Sử dụng mean và std đã lưu từ quá trình training
normalized_value = (value - mean) / std
```

**Mean và Std được lưu trong model:**
- Lưu trong `model_dict["numeric_stats"]`
- Được tính từ dataset gốc để đảm bảo tính nhất quán

### 5.2. Mã hóa phân loại:

```python
# Chuẩn hóa giá trị (lowercase)
normalized_value = str(value).lower()

# One-hot encoding theo mapping đã lưu
# Ví dụ: Gender="Female" → Gender_Female=1, Gender_Male=0
```

**Mapping được lưu trong model:**
- Lưu trong `model_dict["categorical_mapping"]`
- Đảm bảo thứ tự và tên cột nhất quán với dữ liệu training

---

## 6. CÁC VẤN ĐỀ ĐÃ XỬ LÝ

### 6.1. ✅ Xử lý giá trị thiếu:
- **Trạng thái**: Không có giá trị thiếu trong dataset gốc
- **Không cần xử lý**

### 6.2. ✅ Xử lý giá trị ngoại lai:
- **Phương pháp**: Loại bỏ các mẫu có giá trị không hợp lệ (24 mẫu)
- **Tiêu chí**: BMI quá cao/thấp hoặc giá trị không nhất quán

### 6.3. ✅ Chuẩn hóa thang đo:
- **Phương pháp**: Z-score normalization cho tất cả cột số
- **Kết quả**: Tất cả đặc trưng số có cùng thang đo (mean=0, std=1)

### 6.4. ✅ Mã hóa dữ liệu phân loại:
- **Phương pháp**: One-Hot Encoding
- **Kết quả**: Chuyển đổi thành dữ liệu số nhị phân

### 6.5. ✅ Tính toán đặc trưng mới:
- **BMI**: Được tính từ Height và Weight
- **Lý do**: BMI là chỉ số quan trọng để đánh giá béo phì

---

## 7. THỐNG KÊ SAU TIỀN XỬ LÝ

### 7.1. Số lượng mẫu:
- **Trước**: 2,111 mẫu
- **Sau**: 2,087 mẫu (loại bỏ 24 mẫu không hợp lệ)

### 7.2. Số lượng đặc trưng:
- **Trước**: 16 đặc trưng (không tính target)
- **Sau**: 33 đặc trưng (9 số + 23 one-hot + 1 BMI)

### 7.3. Phân phối dữ liệu số sau chuẩn hóa:
```
Mean ≈ 0
Std ≈ 1
Min ≈ -3 đến -2 (tùy cột)
Max ≈ 2 đến 3 (tùy cột)
```

### 7.4. Phân phối dữ liệu phân loại:
- Mỗi cột one-hot chỉ có giá trị `0` hoặc `1` (hoặc `False`/`True`)
- Tổng các cột one-hot trong cùng một nhóm = 1 (ví dụ: Gender_Female + Gender_Male = 1)

---

## 8. LƯU Ý QUAN TRỌNG

### 8.1. Tính nhất quán:
- ⚠️ **Bắt buộc**: Dữ liệu inference phải được chuẩn hóa bằng cùng mean/std và mapping như training
- ⚠️ **Bắt buộc**: Thứ tự và tên cột phải giống hệt với dữ liệu training

### 8.2. Xử lý giá trị mới:
- ⚠️ Nếu có giá trị phân loại mới không có trong training data → cần xử lý đặc biệt
- ⚠️ Giá trị số ngoài phạm vi training → vẫn có thể chuẩn hóa nhưng có thể ảnh hưởng đến độ chính xác

### 8.3. Lưu trữ thông tin chuẩn hóa:
- ✅ Mean/Std của các cột số được lưu trong `models/obesity/final_model.pkl`
- ✅ Mapping của các cột phân loại được lưu trong cùng file
- ✅ Đảm bảo tính nhất quán giữa training và inference

---

## 9. CODE MẪU

### 9.1. Tiền xử lý dữ liệu (Python):

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu gốc
df = pd.read_csv('data/ObesityDataSet.csv')

# 1. Tính BMI
df['BMI'] = df['Weight'] / (df['Height'] ** 2)

# 2. Chuẩn hóa cột số
numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI']
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 3. One-Hot Encoding cột phân loại
categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 
                    'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)

# 4. Lưu dữ liệu đã tiền xử lý
df_encoded.to_csv('data/obesity_preprocessed.csv', index=False)
```

### 9.2. Chuẩn hóa dữ liệu inference:

```python
# Sử dụng mean/std đã lưu từ training
def normalize_features(input_df, numeric_stats):
    df = input_df.copy()
    
    # Tính BMI
    if 'BMI' not in df.columns:
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    
    # Chuẩn hóa cột số
    for col in numeric_stats:
        mean = numeric_stats[col]['mean']
        std = numeric_stats[col]['std'] or 1.0
        df[col] = (df[col] - mean) / std
    
    return df
```

---

## 10. KẾT LUẬN

Quy trình tiền xử lý và chuẩn hóa dữ liệu `ObesityDataSet.csv` bao gồm:

1. ✅ **Tính toán đặc trưng mới**: BMI từ Height và Weight
2. ✅ **Chuẩn hóa dữ liệu số**: Z-score normalization cho 9 cột số
3. ✅ **Mã hóa dữ liệu phân loại**: One-Hot Encoding cho 8 cột phân loại (tạo ra 23 cột)
4. ✅ **Xử lý dữ liệu không hợp lệ**: Loại bỏ 24 mẫu

**Kết quả**: Dataset từ 2,111 mẫu × 17 cột → 2,087 mẫu × 34 cột, sẵn sàng cho việc huấn luyện mô hình Random Forest Classifier.

**Lợi ích**:
- ✅ Cải thiện hiệu suất mô hình
- ✅ Đảm bảo tính nhất quán giữa training và inference
- ✅ Dễ dàng mở rộng và bảo trì

