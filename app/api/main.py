# app/api/main.py
from fastapi import APIRouter

from app.api.routes.statements import router as statements_router

api_router = APIRouter(tags=['default'])


@api_router.get('/health', tags=['default'])
def health():
	return {'status': 'ok'}


# รวม router ของ statements ให้อยู่ใต้ /statements
api_router.include_router(statements_router, prefix='/statements', tags=['Statements'])
