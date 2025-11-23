# PHÂN TÍCH BỘ DỮ LIỆU FOOD_NUTRITION_DATA_FINAL.CSV

## 1. TỔNG QUAN DỮ LIỆU

- **Tổng số mẫu**: 851 thực phẩm (852 dòng bao gồm header)
- **Số cột**: 9 cột
- **Mục đích**: Bảng tra cứu dinh dưỡng thực phẩm Việt Nam

## 2. CẤU TRÚC DỮ LIỆU

### Các cột trong dataset:

1. **Tên thực phẩm** (string): Tên món ăn/thực phẩm bằng tiếng Việt
2. **Nước** (float): Hàm lượng nước (g/100g)
3. **Năng lượng** (float): Calorie (kcal/100g) - cột quan trọng cho calorie lookup
4. **Protein** (float): Hàm lượng đạm (g/100g)
5. **Lipid** (float): Hàm lượng chất béo (g/100g)
6. **Glucid** (float): Hàm lượng carbohydrate (g/100g)
7. **Celluloza** (float): Hàm lượng chất xơ (g/100g)
8. **Tro** (float): Hàm lượng tro/khoáng chất (g/100g)
9. **Link** (string): URL hình ảnh minh họa

## 3. PHÂN TÍCH CHI TIẾT

### 3.1. Phạm vi giá trị dinh dưỡng (từ mẫu dữ liệu):

**Năng lượng (kcal/100g):**
- Tối thiểu: 0 (nước khoáng, rượu)
- Tối đa: ~900 (dầu ăn các loại)
- Phổ biến: 100-400 kcal/100g

**Protein (g/100g):**
- Tối thiểu: 0 (dầu, đường)
- Tối đa: ~75 (tôm khô, cá khô)
- Phổ biến: 0-25 g/100g

**Lipid (g/100g):**
- Tối thiểu: 0 (rau củ, trái cây)
- Tối đa: ~100 (dầu ăn)
- Phổ biến: 0-20 g/100g

**Glucid (g/100g):**
- Tối thiểu: 0 (thịt, cá tươi)
- Tối đa: ~99 (đường kính)
- Phổ biến: 0-80 g/100g

### 3.2. Phân loại thực phẩm (theo mẫu):

1. **Ngũ cốc & Tinh bột**: Gạo, bánh mì, bánh phở, mì, bột các loại
2. **Rau củ**: Khoai lang, khoai tây, củ cải, rau các loại
3. **Đậu & Hạt**: Đậu nành, đậu xanh, lạc, vừng, hạt điều
4. **Thịt & Cá**: Thịt bò, thịt lợn, thịt gà, cá các loại, hải sản
5. **Trái cây**: Chuối, cam, táo, xoài, dưa hấu...
6. **Sữa & Sản phẩm từ sữa**: Sữa tươi, sữa chua, phô mai
7. **Đồ uống**: Nước, trà, cà phê, nước ép
8. **Món ăn chế biến**: Phở, bún, cơm, canh, chè...

### 3.3. Đặc điểm nổi bật:

- **Đa dạng**: Bao gồm cả nguyên liệu thô và món ăn đã chế biến
- **Địa phương hóa**: Nhiều món ăn đặc trưng Việt Nam (phở, bún, chè...)
- **Đầy đủ thông tin**: Có cả hình ảnh minh họa (Link)
- **Đơn vị chuẩn**: Tất cả đều tính theo 100g (trừ một số món có ghi chú)

## 4. SỬ DỤNG TRONG HỆ THỐNG

### 4.1. Trong `calorie_inference.py`:

- **Mục đích**: Tra cứu calorie và thông tin dinh dưỡng
- **Cách sử dụng**:
  1. Người dùng nhập câu hỏi về món ăn
  2. Hệ thống parse và tìm kiếm fuzzy match trong dataset
  3. Trả về calorie và các thông tin dinh dưỡng khác

### 4.2. Quy trình lookup:

```
Input: "Tôi ăn 1 bát phở bò"
  ↓
Parse: food="phở bò", quantity=1, unit="bát"
  ↓
Fuzzy match với "Bánh phở" hoặc "Phở bò tái" trong dataset
  ↓
Lấy Năng lượng (kcal/100g) từ dataset
  ↓
Tính: calories = (Năng lượng * grams) / 100
  ↓
Output: Calorie ước tính + thông tin dinh dưỡng
```

## 5. ĐIỂM MẠNH

✅ **Phong phú**: 851 món ăn/thực phẩm  
✅ **Chuẩn hóa**: Đơn vị thống nhất (g/100g, kcal/100g)  
✅ **Đầy đủ**: Có đủ các thành phần dinh dưỡng chính  
✅ **Tiếng Việt**: Tên món ăn bằng tiếng Việt, phù hợp với hệ thống  
✅ **Có hình ảnh**: Link ảnh minh họa hỗ trợ UI  

## 6. HẠN CHẾ & KHUYẾN NGHỊ

### 6.1. Hạn chế:

⚠️ **Thiếu thông tin khẩu phần**: Chỉ có giá trị /100g, không có khẩu phần chuẩn  
⚠️ **Không có biến thể**: Một số món có nhiều cách chế biến nhưng chỉ có 1 entry  
⚠️ **Thiếu món mới**: Có thể thiếu các món ăn hiện đại, fusion  

### 6.2. Khuyến nghị cải thiện:

1. **Thêm cột "Khẩu phần chuẩn"**: Ví dụ "1 bát phở = 250g"
2. **Thêm cột "Loại"**: Phân loại rõ ràng (ngũ cốc, thịt, rau...)
3. **Cập nhật định kỳ**: Thêm món ăn mới, món fusion
4. **Chuẩn hóa tên**: Đảm bảo tên món ăn nhất quán (ví dụ: "phở bò" vs "bánh phở")

## 7. THỐNG KÊ MẪU

### Top thực phẩm năng lượng cao:
- Dầu ăn các loại: ~900 kcal/100g
- Mỡ lợn: ~827 kcal/100g
- Bơ: ~756 kcal/100g

### Top thực phẩm năng lượng thấp:
- Nước khoáng: 0 kcal/100g
- Rau xanh các loại: 14-30 kcal/100g
- Dưa chuột: 16 kcal/100g

### Top thực phẩm giàu protein:
- Tôm khô: ~75.6 g/100g
- Cá khô: ~43-60 g/100g
- Thịt bò khô: ~51 g/100g

## 8. KẾT LUẬN

Bộ dữ liệu `food_nutrition_data_final.csv` là một nguồn tài nguyên tốt cho hệ thống calorie lookup, với:
- **Độ phủ rộng**: 851 món ăn/thực phẩm
- **Độ chính xác**: Dữ liệu dinh dưỡng chuẩn
- **Tính thực tế**: Bao gồm cả món ăn Việt Nam phổ biến

Hệ thống hiện tại sử dụng fuzzy matching để tìm kiếm, giúp xử lý được các biến thể tên món ăn và lỗi chính tả của người dùng.



