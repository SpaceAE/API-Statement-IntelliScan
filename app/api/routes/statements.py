from typing import Annotated

from fastapi import APIRouter, Form, HTTPException

from app.core.file import (
	IncorrectPasswordException,
	PasswordRequiredException,
	read_file,
)
from app.models.request import PredictForm
from app.models.response import PredictResponse

router = APIRouter(
	prefix='/statements',
	tags=['Statements'],
)


@router.post(
	'/predict',
	summary='Analyze Statement Risk',
	description='Analyze and classify statement documents for risk assessment',
	responses={
		200: {
			'description': 'Successful Response',
			'content': {
				'application/json': {
					'example': {'prediction': 'fraud', 'confidence': 0.91}
				}
			},
		},
		400: {
			'description': 'Bad Request',
			'content': {
				'application/json': {
					'examples': {
						'invalid_file_type': {
							'summary': 'Invalid file type',
							'value': {
								'message': (
									'Invalid file type. Only PDF files are accepted.'
								)
							},
						},
					}
				}
			},
		},
		403: {
			'description': 'Forbidden',
			'content': {
				'application/json': {
					'examples': {
						'incorrect_password': {
							'summary': 'Incorrect Password',
							'value': {
								'message': 'Incorrect password for the encrypted PDF.'
							},
						},
					}
				}
			},
		},
		422: {
			'description': 'Unprocessable Entity',
			'content': {
				'application/json': {
					'examples': {
						'missing_required_field': {
							'summary': 'Missing Required Field',
							'value': {
								'message': 'Missing required fields',
								'errors': 'Required body (file)',
							},
						},
						'password_required': {
							'summary': 'Password Required',
							'value': {
								'message': (
									'Password is required for this encrypted PDF.'
								)
							},
						},
					}
				}
			},
		},
		500: {
			'description': 'Internal Server Error',
			'content': {
				'application/json': {
					'example': {
						'message': 'Internal server error',
					}
				}
			},
		},
	},
)
async def predict(
	form: Annotated[PredictForm, Form(media_type='multipart/form-data')],
) -> PredictResponse:
	file, password = form.file, form.password
	if (
		file.content_type != 'application/pdf'
		or file.filename.split('.')[-1].lower() != 'pdf'
	):
		raise HTTPException(
			status_code=400,
			detail={'message': 'Invalid file type. Only PDF files are accepted.'},
		)

	try:
		read_file(file.file, password)

		# Dummy prediction logic for demonstration purposes
		return PredictResponse(
			prediction='normal',
			confidence=0.95,
		)
	except IncorrectPasswordException:
		raise HTTPException(
			status_code=403,
			detail={'message': 'Incorrect password for the encrypted PDF.'},
		)
	except PasswordRequiredException:
		raise HTTPException(
			status_code=422,
			detail={'message': 'Password is required for this encrypted PDF.'},
		)
