import os
import traceback

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

APP_DEBUG = os.getenv('APP_DEBUG', '1') in ('1', 'true', 'True')


async def http_exception_handler(_: Request, exc: HTTPException):
	# ส่ง detail เดิมออกไปตรง ๆ เพื่อให้ Swagger เห็นตามที่ route โยนมา
	detail = (
		exc.detail if isinstance(exc.detail, dict) else {'message': str(exc.detail)}
	)
	return JSONResponse(status_code=exc.status_code, content=detail)


async def general_exception_handler(_: Request, exc: Exception):
	# ไม่เขียนไฟล์ ไม่ print เพิ่ม — ส่งรายละเอียดกลับไปที่ Swagger เฉพาะตอน dev
	payload = {'message': 'Internal server error'}
	if APP_DEBUG:
		payload['traceback'] = traceback.format_exc().splitlines()
		payload['code'] = 'UNHANDLED_EXCEPTION'
	return JSONResponse(
		status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=payload
	)
