from typing import Dict, List

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
	missing_fields: Dict[str, List[str]] = {}

	for error in exc.errors():
		if error['type'] == 'missing':
			field_path = error['loc']
			[parent_field, missing_field] = field_path
			missing_fields.setdefault(parent_field, []).append(missing_field)

	if not missing_fields:
		error_message = 'Validation error occurred'
	else:
		field_descriptions = [
			f'{field} ({", ".join(fields)})' for field, fields in missing_fields.items()
		]
		error_message = f'Required {" ".join(field_descriptions)}'

	return JSONResponse(
		status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
		content={
			'message': 'Missing required fields',
			'errors': error_message,
		},
	)
