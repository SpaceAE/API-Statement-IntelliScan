from fastapi import APIRouter

from app.api.routes import statements

api_router = APIRouter()
api_router.include_router(statements.router)
