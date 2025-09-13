from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.main import api_router
from .core.config import settings
from .core.model import get_model


@asynccontextmanager
async def lifespan(_: FastAPI):
	get_model()
	yield


app = FastAPI(
	title=settings.PROJECT_NAME,
	openapi_url=f'{settings.API_PREFIX}/openapi.json',
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


app.include_router(api_router, prefix=settings.API_PREFIX)
