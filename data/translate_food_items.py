import os
import pandas as pd
from tqdm import tqdm

try:
    # deep-translator is compatible with newer Python versions
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

# ====== CẤU HÌNH ======
# Use paths relative to this script file so the script can be run from any CWD
HERE = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(HERE, "FOOD-DATA.csv")        # File gốc
output_file = os.path.join(HERE, "daily_food_nutrition_dataset_vi.csv")    # File xuất ra

# ====== CHƯƠNG TRÌNH CHÍNH ======
def translate_food_items():
    # Đọc file CSV
    print("🔹 Đang đọc dữ liệu...")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"❌ Không tìm thấy file input: {input_file}\nMake sure the CSV is located in the same folder as this script or update `input_file` path.")

    df = pd.read_csv(input_file)

    # Kiểm tra cột tồn tại
    if "Food_Item" not in df.columns:
        raise ValueError("❌ Không tìm thấy cột 'Food_Item' trong file CSV!")

    # Khởi tạo translator
    if GoogleTranslator is None:
        raise ImportError("deep-translator is not installed. Install it with: python -m pip install deep-translator")

    translator = GoogleTranslator(source='auto', target='vi')

    # Dịch từng dòng
    translated = []
    print("🔹 Đang dịch sang tiếng Việt (có thể mất vài phút)...")
    for item in tqdm(df["Food_Item"], desc="Translating"):
        try:
            text_vi = translator.translate(item)
        except Exception as e:
            print(f"Lỗi dịch '{item}': {e}")
            text_vi = item  # Giữ nguyên nếu lỗi
        translated.append(text_vi)

    # Gán lại cột Food_Item
    df["Food_Item"] = translated

    # Xuất file CSV mới
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"✅ Hoàn tất! File đã được lưu thành: {output_file}")


if __name__ == "__main__":
    translate_food_items()
