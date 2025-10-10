from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

from services.infer import infer

app = FastAPI(title="Nutrition Inference API")


class PredictRequest(BaseModel):
    content: str


@app.post("/predict")
async def predict(req: PredictRequest) -> Dict[str, Any]:
    text = req.content
    if not text or not isinstance(text, str):
        raise HTTPException(status_code=400, detail="Field 'content' is required and must be a string")
    try:
        result = infer(text)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
