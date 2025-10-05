import os

# routes
from fastapi import APIRouter, FastAPI, HTTPException

from app.api import error_handlers as eh
from app.api.routes.statements import router as statements_router

APP_DEBUG = os.getenv('APP_DEBUG', '1') in ('1', 'true', 'True')

app = FastAPI(
	title='API-Statement-IntelliScan',
	version='0.1.0',
	openapi_url='/openapi.json',
	docs_url='/docs',
	redoc_url=None,
	debug=APP_DEBUG,  # ให้ Starlette แสดง traceback ในคอนโซล (ไม่บังคับ)
)


api_router = APIRouter(tags=['default'])


@app.get('/api/v1/health', tags=['default'])
def health():
	return {'status': 'ok'}


# include
app.include_router(api_router)
app.include_router(statements_router, prefix='/api/v1')

# handlers (ไม่เขียนไฟล์)
app.add_exception_handler(HTTPException, eh.http_exception_handler)
app.add_exception_handler(Exception, eh.general_exception_handler)
