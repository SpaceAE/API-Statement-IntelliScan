import io
import re
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Form, HTTPException, Query

from app.core.config import settings
from app.core.file import (
	IncorrectPasswordException,
	PasswordRequiredException,
	read_file,
)
from app.core.model import predict_proba_df, summarize_document
from app.core.pdfToCSV import (
	StatementFormatUnsupported,
	_detect_bank_brand,
	extract_statement_df,
)
from app.models.request import PredictForm, PredictQueryParam
from app.models.response import PredictResponse, TransactionResult

router = APIRouter(
	prefix='/statements',
	tags=['Statements'],
)

_REQUIRED_COLS = [
	'tx_datetime',
	'code_channel_raw',
	'debit_amount',
	'credit_amount',
	'balance_amount',
	'description_text',
]

_ALLOWED_CT = {
	'text/csv',
	'application/csv',
	'application/vnd.ms-excel',
	'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
}


def _norm_col(s: str) -> str:
	s = (s or '').strip().lower()
	s = re.sub(r'\s+', '_', s)
	s = re.sub(r'[^a-z0-9_]+', '', s)
	return s


def _read_tb_file_to_df(filename: str, content_type: str, raw: bytes) -> pd.DataFrame:
	fname = (filename or '').lower()
	ext = fname.rsplit('.', 1)[-1] if '.' in fname else ''
	bio = io.BytesIO(raw)

	try:
		if ext == 'csv' or content_type in {'text/csv', 'application/csv'}:
			df = pd.read_csv(bio)
		elif ext in {'xlsx', 'xls'} or content_type in _ALLOWED_CT:
			df = pd.read_excel(bio, engine=None)
		else:
			bio.seek(0)
			df = pd.read_csv(bio)
	except Exception as e:
		raise HTTPException(
			status_code=400,
			detail={
				'message': f'Failed to read file: {e}',
				'allowed_content_types': sorted(_ALLOWED_CT | {'application/pdf'}),
			},
		)

	df.columns = [_norm_col(str(c)) for c in df.columns]

	missing = [c for c in _REQUIRED_COLS if c not in df.columns]
	if missing:
		raise HTTPException(
			status_code=422,
			detail={
				'message': 'Missing required columns',
				'missing': missing,
				'required': _REQUIRED_COLS,
			},
		)
	return df


def _coerce_and_format(df: pd.DataFrame) -> pd.DataFrame:
	out = df[_REQUIRED_COLS].copy()
	for col in ['debit_amount', 'credit_amount', 'balance_amount']:
		out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0.0)

	dt_parsed = pd.to_datetime(out['tx_datetime'], errors='coerce', dayfirst=True)
	mask = dt_parsed.notna()
	out.loc[mask, 'tx_datetime'] = dt_parsed[mask].dt.strftime('%d/%m/%Y %H:%M')
	out['description_text'] = out['description_text'].astype(str).fillna('')
	return out


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
								'message': {
									'Invalid file type. Only PDF files are accepted.'
								}
							},
						},
						'statement_format_unsupported': {
							'summary': 'Statement format is not supported',
							'value': {
								'message': 'This statement format is not supported. '
								'Please upload statements from SCB.'
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
						'validation_error': {
							'summary': 'Validation error',
							'value': {
								'message': 'Validation error',
								'errors': [
									'body -> file: Value error, Expected UploadFile, '
									"received: <class 'str'>"
								],
							},
						},
						'missing_required_field': {
							'summary': 'Missing Required Field',
							'value': {
								'message': 'Validation error',
								'errors': ['Missing required field: body -> file'],
							},
						},
						'password_required': {
							'summary': 'Password Required',
							'value': {
								'message': {
									'Password is required for this encrypted PDF.'
								}
							},
						},
					}
				}
			},
		},
		500: {
			'description': 'Internal Server Error',
			'content': {
				'application/json': {'example': {'message': 'Internal server error'}}
			},
		},
	},
)
async def predict(
	form: Annotated[PredictForm, Form(media_type='multipart/form-data')],
	query: Annotated[PredictQueryParam, Query()],
) -> PredictResponse:
	file, password = form.file, form.password
	fname = (file.filename or '').lower()
	ext = fname.rsplit('.', 1)[-1] if '.' in fname else ''
	ctype = (file.content_type or '').lower()

	try:
		# อ่านไฟล์เป็น bytes + ตรวจ PDF encryption/password
		raw = read_file(file.file, password)

		# ตัดสินชนิดไฟล์
		is_pdf = ext == 'pdf' or ctype == 'application/pdf'
		is_table = ext in {'csv', 'xlsx', 'xls'} or ctype in _ALLOWED_CT

		if is_pdf and not is_table:
			bio = io.BytesIO(raw)
			df, _meta = extract_statement_df(bio, password=password, return_meta=True)
			if df.empty:
				raise HTTPException(
					status_code=422,
					detail={'message': 'No transactions detected in the PDF.'},
				)

			# ตรวจสอบฟอร์แมตของ statement โดยใช้ _detect_bank_brand
			brand = _detect_bank_brand(file.file, password=password)
			if brand != 'scb':
				raise HTTPException(
					status_code=400,
					detail={
						'message': 'This statement format is not supported. '
						'Please upload statements from SCB.'
					},
				)

		elif is_table and not is_pdf:
			df = _read_tb_file_to_df(fname, ctype, raw)

		elif is_pdf and is_table:
			raise HTTPException(
				status_code=400,
				detail={'message': 'Ambiguous file type (both table & PDF detected).'},
			)
		else:
			raise HTTPException(
				status_code=400,
				detail={
					'message': 'Invalid file type. Upload a CSV/XLSX or PDF.',
					'allowed_content_types': sorted(_ALLOWED_CT | {'application/pdf'}),
				},
			)

		# ตรวจสอบการขาดหายของคอลัมน์ที่จำเป็น
		missing = [c for c in _REQUIRED_COLS if c not in df.columns]
		if missing:
			raise HTTPException(
				status_code=422,
				detail={'message': f'Missing required columns: {missing}'},
			)

		# ตรวจสอบให้แน่ใจว่า DataFrame มีข้อมูลที่ต้องการ
		if df.empty:
			raise HTTPException(
				status_code=422,
				detail={'message': 'No valid transactions in the provided data.'},
			)

		# preprocessing
		df_model = _coerce_and_format(df)

		# predict
		probas = predict_proba_df(df_model)

		threshold = settings.MODEL_THRESHOLD
		summary = summarize_document(probas, threshold=threshold)

		# build items
		items: list[TransactionResult] = []
		for i, row in df_model.reset_index(drop=True).iterrows():
			p = float(probas.loc[i])
			items.append(
				TransactionResult(
					idx=int(i),
					tx_datetime=str(row.tx_datetime),
					code_channel_raw=str(row.code_channel_raw),
					debit_amount=float(row.debit_amount),
					credit_amount=float(row.credit_amount),
					balance_amount=float(row.balance_amount),
					description_text=str(row.description_text),
					fraud_score=round(p, 4),
					is_fraud=bool(p >= threshold),
				)
			)

		fraud_items = [tx for tx in items if tx.is_fraud]
		transactions_total = len(df_model)
		fraud_count = len(fraud_items)

		# ใช้ query.only_fraud
		transactions_out = fraud_items if query.only_fraud else items

		return PredictResponse(
			prediction=summary['prediction'],
			confidence=summary['confidence'],
			fraud_count=fraud_count,
			transactions_count=transactions_total,
			transactions=transactions_out,
		)

	except IncorrectPasswordException:
		raise HTTPException(
			status_code=403,
			detail={'message': 'Incorrect password for the encrypted PDF.'},
		)
	except PasswordRequiredException:
		raise HTTPException(
			status_code=403,
			detail={'message': 'Password is required for this encrypted PDF.'},
		)
	except StatementFormatUnsupported as e:
		raise HTTPException(
			status_code=400,
			detail={'message': str(e)},
		)
	except KeyError as e:
		raise HTTPException(
			status_code=500, detail={'message': f'KeyError occurred: {str(e)}'}
		)
	except Exception as e:
		raise HTTPException(
			status_code=500, detail={'message': f'Internal server error: {str(e)}'}
		)
