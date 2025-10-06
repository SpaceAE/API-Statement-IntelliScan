# app/api/routes/statements.py
import io
import os
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.core.file import (
	IncorrectPasswordException,
	PasswordRequiredException,
	read_file,
)
from app.core.model import predict_proba_df, summarize_document
from app.core.pdf_to_csv import (
	# ← เพิ่มสอง Exception ด้านล่าง (มาจากไฟล์ pdf_to_csv.py ที่เราเพิ่ม brand gate)
	StatementFormatUnsupported,
	StatementParsingFailed,
	extract_statement_df,
)
from app.models.request import PredictForm
from app.models.response import (
	PredictResponseWithDetails,
	TxResult,
)

APP_DEBUG = os.getenv('APP_DEBUG', '1') in ('1', 'true', 'True')
router = APIRouter(prefix='/statements', tags=['Statements'])


@router.post(
	'/predict',
	summary='Analyze Statement Risk',
	description=(
		'Unlocks the PDF (if encrypted), '
		'extracts transactions into a normalized DataFrame, '
		'converts the rows to CSV in-memory for a consistent model '
		'input format, reloads it, scores each transaction with the '
		'fraud model, and returns a JSON summary + per-row details.'
	),
	responses={
		200: {'description': 'OK'},
		400: {'description': 'Invalid file'},
		403: {'description': 'Wrong password'},
		422: {'description': 'No transactions / Validation'},
		500: {'description': 'Internal error'},
	},
)
async def predict(
	form: Annotated[PredictForm, Depends(PredictForm.as_form)],
) -> PredictResponseWithDetails:
	file, password = form.file, form.password

	if (file.content_type != 'application/pdf') or (
		file.filename.split('.')[-1].lower() != 'pdf'
	):
		raise HTTPException(
			status_code=400,
			detail={'message': 'Invalid file type. Only PDF files are accepted.'},
		)

	try:
		# ลองเปิด/ปลดล็อกไฟล์ก่อน (อาจโยน IncorrectPassword/PasswordRequired)
		read_file(file.file, password)
		file.file.seek(0)

		# ดึงข้อมูลพร้อม meta (เพื่อ debug)
		df, meta = extract_statement_df(file.file, password=password, return_meta=True)  # type: ignore
		if df.empty:
			detail = {
				'message': 'No transactions detected in the PDF.',
				'code': 'NO_TRANSACTIONS',
			}
			if APP_DEBUG:
				detail['debug'] = meta
			raise HTTPException(status_code=422, detail=detail)

		required_cols = [
			'tx_datetime',
			'code_channel_raw',
			'debit_amount',
			'credit_amount',
			'balance_amount',
			'description_text',
		]
		missing = [c for c in required_cols if c not in df.columns]
		if missing:
			raise HTTPException(
				status_code=422,
				detail={
					'message': f'Missing required columns: {missing}',
					'code': 'MISSING_COLUMNS',
				},
			)

		df_model = df[required_cols].copy()
		for num_col in ['debit_amount', 'credit_amount', 'balance_amount']:
			df_model[num_col] = pd.to_numeric(
				df_model[num_col], errors='coerce'
			).fillna(0.0)

		probas = predict_proba_df(df_model)
		threshold = float(os.getenv('MODEL_THRESHOLD', '0.5'))
		summary = summarize_document(probas, threshold=threshold)

		items = []
		for i, row in df_model.reset_index(drop=True).iterrows():
			p = float(probas.loc[i])
			items.append(
				TxResult(
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

		return PredictResponseWithDetails(
			prediction=summary['prediction'],
			confidence=summary['confidence'],
			threshold=summary['threshold'],
			fraud_count=summary['fraud_count'],
			total=summary['total'],
			transactions=items,
		)

	except StatementFormatUnsupported as e:
		# ไม่ใช่ SCB หรือรูปแบบไม่รองรับ
		detail = {'message': str(e), 'code': 'UNSUPPORTED_BANK'}
		if APP_DEBUG:
			detail['hint'] = 'Only SCB statements are supported.'
		raise HTTPException(status_code=422, detail=detail)

	except StatementParsingFailed as e:
		# ตรวจพบว่าเป็น SCB แต่ parse ไม่ได้
		detail = {'message': str(e), 'code': 'PARSE_FAILED'}
		if APP_DEBUG:
			# แนบข้อมูลช่วย debug เพิ่มเติม
			detail.setdefault('debug', {}).update(
				{'note': 'SCB brand detected but no rows found'}
			)
		raise HTTPException(status_code=422, detail=detail)

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
	except Exception as e:
		raise HTTPException(
			status_code=500,
			detail={'message': 'Internal server error', 'error': str(e)},
		)


@router.post(
	'/predict:csv',
	summary='Analyze Statement Risk and return CSV',
	description=(
		'Upload an encrypted PDF statement, extract transactions, '
		'score fraud risk, and download as CSV.'
	),
)
async def predict_csv(form: Annotated[PredictForm, Depends(PredictForm.as_form)]):
	file, password = form.file, form.password

	if (file.content_type != 'application/pdf') or (
		file.filename.split('.')[-1].lower() != 'pdf'
	):
		raise HTTPException(
			status_code=400,
			detail={'message': 'Invalid file type. Only PDF files are accepted.'},
		)

	try:
		read_file(file.file, password)
		file.file.seek(0)

		# ใช้ return_meta เพื่อ debug ได้ด้วยถ้าจำเป็น
		df, meta = extract_statement_df(file.file, password=password, return_meta=True)  # type: ignore
		if df.empty:
			detail = {
				'message': 'No transactions detected in the PDF.',
				'code': 'NO_TRANSACTIONS',
			}
			if APP_DEBUG:
				detail['debug'] = meta
			raise HTTPException(status_code=422, detail=detail)

		required_cols = [
			'tx_datetime',
			'code_channel_raw',
			'debit_amount',
			'credit_amount',
			'balance_amount',
			'description_text',
		]
		missing = [c for c in required_cols if c not in df.columns]
		if missing:
			raise HTTPException(
				status_code=422,
				detail={
					'message': f'Missing required columns: {missing}',
					'code': 'MISSING_COLUMNS',
				},
			)

		df_model = df[required_cols].copy()
		for num_col in ['debit_amount', 'credit_amount', 'balance_amount']:
			df_model[num_col] = pd.to_numeric(
				df_model[num_col], errors='coerce'
			).fillna(0.0)

		probas = predict_proba_df(df_model)
		threshold = float(os.getenv('MODEL_THRESHOLD', '0.5'))

		scored_df = df_model.copy()
		scored_df['fraud_score'] = probas.astype(float).round(6)
		scored_df['is_fraud'] = (probas >= threshold).astype(bool)

		csv_bytes = scored_df.to_csv(index=False).encode('utf-8')
		fname_base = (file.filename or 'statement.pdf').rsplit('.', 1)[0]
		fname = f'{fname_base}_predicted.csv'

		return StreamingResponse(
			io.BytesIO(csv_bytes),
			media_type='text/csv',
			headers={'Content-Disposition': f'attachment; filename="{fname}"'},
		)

	except StatementFormatUnsupported as e:
		detail = {'message': str(e), 'code': 'UNSUPPORTED_BANK'}
		if APP_DEBUG:
			detail['hint'] = 'Only SCB statements are supported.'
		raise HTTPException(status_code=422, detail=detail)

	except StatementParsingFailed as e:
		detail = {'message': str(e), 'code': 'PARSE_FAILED'}
		if APP_DEBUG:
			detail.setdefault('debug', {}).update(
				{'note': 'SCB brand detected but no rows found'}
			)
		raise HTTPException(status_code=422, detail=detail)

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
	except Exception as e:
		raise HTTPException(
			status_code=500,
			detail={'message': 'Internal server error', 'error': str(e)},
		)
