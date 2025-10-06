# app/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.api.main import api_router
from app.middlewares.error_handling import (
	general_exception_handler,
	http_exception_handler,
	validation_exception_handler,
)

try:
	from .core.config import settings

	API_PREFIX = getattr(settings, 'API_PREFIX', '/api/v1')
	PROJECT_NAME = getattr(settings, 'PROJECT_NAME', 'API-Statement-IntelliScan')
	ENVIRONMENT = getattr(settings, 'ENVIRONMENT', 'development')
except Exception:
	API_PREFIX = '/api/v1'
	PROJECT_NAME = 'API-Statement-IntelliScan'
	ENVIRONMENT = 'development'


@asynccontextmanager
async def lifespan(_: FastAPI):
	if ENVIRONMENT == 'production':
		try:
			from .core.model import get_model

			get_model()
		except Exception:
			# กัน import พังตอน dev
			pass
	yield


app = FastAPI(
	title=PROJECT_NAME,
	version='0.1.0',
	lifespan=lifespan,
	openapi_url='/openapi.json',
	docs_url='/docs',
	redoc_url=None,
)


app.add_middleware(
	CORSMiddleware,
	allow_origins=['*'],
	allow_credentials=True,
	allow_methods=['*'],
	allow_headers=['*'],
)


app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)


# รวมทุก API router ใต้ /api/v1
app.include_router(api_router, prefix=API_PREFIX)
