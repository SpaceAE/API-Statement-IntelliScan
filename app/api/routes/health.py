from fastapi import APIRouter

from app.models.response import HealthResponse

router = APIRouter(
	prefix='/health',
	tags=['Health'],
)


@router.get('', summary='Health Check')
async def health_check() -> HealthResponse:
	return HealthResponse(status='healthy')
