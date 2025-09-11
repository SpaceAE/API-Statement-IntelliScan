from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .configs.env import env
from .routers import predicts

app = FastAPI()

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

app.include_router(predicts.router)


@app.get('/')
def root():
	return {'message': f'server is running {env.message}'}
