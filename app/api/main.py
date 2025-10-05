# app/api/main.py
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# ⬇️ แทนที่จะ: from app.core.model import get_model, predict_one
from app.core import model as model_core

api_router = APIRouter()


class PredictIn(BaseModel):
	tx_datetime: str = Field(..., description='ISO datetime เช่น 2025-10-03T14:15:00Z')
	code_channel_raw: str = Field(..., description='เช่น TRF/SCB EASY, POS/KERRY')
	debit_amount: Optional[float] = 0.0
	credit_amount: Optional[float] = 0.0
	balance_amount: Optional[float] = 0.0
	description_text: Optional[str] = ''


class PredictOut(BaseModel):
	score: float
	label: int
	threshold: float


@api_router.get('/health')
def health():
	try:
		model_core.get_model()
		return {'status': 'ok', 'model': 'loaded'}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f'health check failed: {e}')


@api_router.post('/predict', response_model=PredictOut)
def predict(body: PredictIn):
	try:
		# ถ้าใช้ Pydantic v2: model_dump(); v1: dict()
		payload = body.model_dump() if hasattr(body, 'model_dump') else body.dict()
		result = model_core.predict_one(payload)
		return PredictOut(**result)
	except Exception as e:
		raise HTTPException(status_code=500, detail=f'prediction failed: {e}')
