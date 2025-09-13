from typing import Annotated

from fastapi import APIRouter, File, HTTPException

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
		422: {
			'description': 'Unprocessable Entity',
			'content': {
				'application/json': {
					'examples': {
						'Missing_Required_Field': {
							'summary': 'Missing Required Field',
							'value': {
								'message': 'Missing required fields',
								'errors': 'Required password (statement, password)',
							},
						}
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
async def predict(form: Annotated[PredictForm, File()]) -> PredictResponse:
	if (
		form.file.content_type != 'application/pdf'
		or form.file.filename.split('.')[-1].lower() != 'pdf'
	):
		raise HTTPException(
			status_code=400,
			detail={'message': 'Invalid file type. Only PDF files are accepted.'},
		)

	# Dummy prediction logic for demonstration purposes
	return PredictResponse(
		prediction='normal',
		confidence=0.95,
	)
