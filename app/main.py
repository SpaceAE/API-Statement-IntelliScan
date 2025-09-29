from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.middlewares.error_handling import (
	general_exception_handler,
	http_exception_handler,
	validation_exception_handler,
)

from .api.main import api_router
from .core.config import settings
from .core.model import get_model


@asynccontextmanager
async def lifespan(_: FastAPI):
	if settings.ENVIRONMENT == 'production':
		get_model()
	yield


app = FastAPI(
	title=settings.PROJECT_NAME,
	lifespan=lifespan,
)

origins = [
	'*',
]

app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=['*'],
	allow_headers=['*'],
)

app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)


app.include_router(api_router, prefix=settings.API_PREFIX)
