from typing import List

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
	"""Handle HTTP exceptions and format them into a user-friendly response."""
	return JSONResponse(
		status_code=exc.status_code,
		content={'message': exc.detail.get('message', 'An error occurred')},
	)


async def general_exception_handler(_: Request, exc: Exception) -> JSONResponse:
	"""Handle general exceptions and format them into a user-friendly response."""
	return JSONResponse(
		status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
		content={'message': 'Internal server error'},
	)


async def validation_exception_handler(
	_: Request, exc: RequestValidationError
) -> JSONResponse:
	"""Handle validation errors and format them into a user-friendly response."""
	error_details: List[str] = []
	print(exc.errors())

	for error in exc.errors():
		loc = ' -> '.join(str(x) for x in error.get('loc', []))
		err_type = error.get('type', 'unknown')
		msg = error.get('msg', 'Invalid value')

		if err_type == 'missing':
			error_details.append(f'Missing required field: {loc}')
		else:
			error_details.append(f'{loc}: {msg}')

	return JSONResponse(
		status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
		content={
			'message': 'Validation error',
			'errors': error_details,
		},
	)
