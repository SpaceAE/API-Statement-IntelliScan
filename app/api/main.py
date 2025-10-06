# app/api/main.py
from fastapi import APIRouter

from app.api.routes.statements import router as statements_router

api_router = APIRouter(tags=['default'])


@api_router.get('/health', tags=['default'])
def health():
	return {'status': 'ok'}


api_router.include_router(statements_router)  # ไม่ต้องใส่ prefix ซ้ำ
