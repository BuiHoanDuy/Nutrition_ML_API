import pytest
import httpx
import os

# --- Cấu hình ---
# Giả sử API đang chạy tại địa chỉ localhost:8000
BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Sử dụng httpx.AsyncClient để thực hiện các yêu cầu bất đồng bộ
@pytest.mark.asyncio
async def test_api_is_running():
    """Kiểm tra xem server API có đang hoạt động không."""
    async with httpx.AsyncClient() as client:
        try:
            # Endpoint /docs của FastAPI là một mục tiêu tốt để kiểm tra
            response = await client.get(f"{BASE_URL}/docs")
            # Nếu không có lỗi 2xx, một exception sẽ được ném ra
            response.raise_for_status()
            print(f"\n✅ Server API đang hoạt động tại {BASE_URL}")
        except (httpx.ConnectError, httpx.HTTPStatusError) as e:
            pytest.fail(
                f"❌ Không thể kết nối đến server API tại {BASE_URL}. "
                f"Hãy chắc chắn rằng bạn đã chạy server với lệnh: uvicorn api.main:app --reload\nLỗi: {e}"
            )

@pytest.mark.asyncio
async def test_predict_endpoint():
    """Kiểm thử endpoint /predict để phân tích món ăn và calo."""
    async with httpx.AsyncClient() as client:
        payload = {"content": "tôi ăn 2 quả trứng gà luộc"}
        response = await client.post(f"{BASE_URL}/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "result" in data
        assert data["result"]["found_in_master"] is True
        assert "nutrition" in data["result"]
        print(f"✅ /predict endpoint hoạt động đúng với câu: '{payload['content']}'")

@pytest.mark.asyncio
async def test_recommend_meal_plan_endpoint():
    """Kiểm thử endpoint /recommend_meal_plan với câu hỏi ngôn ngữ tự nhiên."""
    async with httpx.AsyncClient() as client:
        # Câu hỏi này sẽ kiểm tra khả năng phân tích của LLM (nếu có API key) hoặc fallback
        payload = {"question": "Tôi là người bình thường, muốn tăng cơ thì nên ăn gì?"}
        response = await client.post(f"{BASE_URL}/recommend_meal_plan", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "response" in data
        assert isinstance(data["response"], str)
        print(f"✅ /recommend_meal_plan endpoint hoạt động đúng với câu hỏi: '{payload['question']}'")

# @pytest.mark.asyncio
# async def test_generate_daily_menu_endpoint():
#     """Kiểm thử endpoint /generate_daily_menu để tạo thực đơn cả ngày."""
#     async with httpx.AsyncClient() as client:
#         params = {"num_meals": 3}
#         response = await client.get(f"{BASE_URL}/generate_daily_menu", params=params)
        
#         assert response.status_code == 200
#         data = response.json()
#         assert data["success"] is True
#         assert "menu" in data
#         assert "bữa_sáng" in data["menu"]
#         assert "bữa_trưa" in data["menu"]
#         assert "bữa_tối" in data["menu"]
#         print("✅ /generate_daily_menu endpoint hoạt động đúng.")

# @pytest.mark.asyncio
# async def test_suggest_next_meal_endpoint():
#     """Kiểm thử endpoint /suggest_next_meal để gợi ý bữa ăn tiếp theo."""
#     async with httpx.AsyncClient() as client:
#         payload = {"question": "Tôi vừa ăn phở bò xong"}
#         params = {"num_suggestions": 3}
#         response = await client.post(f"{BASE_URL}/suggest_next_meal", json=payload, params=params)
        
#         assert response.status_code == 200
#         data = response.json()
#         assert data["success"] is True
#         assert "suggestions" in data
#         assert len(data["suggestions"]) <= 3
#         print(f"✅ /suggest_next_meal endpoint hoạt động đúng với câu: '{payload['question']}'")
