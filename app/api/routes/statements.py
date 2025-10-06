# app/api/routes/statements.py
import io
import os
import re
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile

from app.core.file import (
	IncorrectPasswordException,
	PasswordRequiredException,
	read_file,
)
from app.core.model import predict_proba_df, summarize_document
from app.core.pdf_to_csv import (
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


def _read_table_file_to_df(file: UploadFile) -> pd.DataFrame:
	fname = (file.filename or '').lower()
	ext = fname.rsplit('.', 1)[-1] if '.' in fname else ''

	content = file.file.read()
	if not content:
		raise HTTPException(status_code=400, detail={'message': 'Empty file'})

	bio = io.BytesIO(content)

	try:
		if ext == 'csv' or (file.content_type in {'text/csv', 'application/csv'}):
			df = pd.read_csv(bio)
		elif ext in {'xlsx', 'xls'} or (
			file.content_type
			in {
				'application/vnd.ms-excel',
				'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
			}
		):
			df = pd.read_excel(bio, engine='openpyxl')
		else:
			bio.seek(0)
			df = pd.read_csv(bio)
	except Exception as e:
		raise HTTPException(
			status_code=400,
			detail={
				'message': f'Failed to read file: {str(e)}',
				'allowed_content_types': sorted(_ALLOWED_CT),
			},
		)

	df = df.copy()
	df.columns = [_norm_col(str(c)) for c in df.columns]

	missing = [c for c in _REQUIRED_COLS if c not in df.columns]
	if missing:
		raise HTTPException(
			status_code=422,
			detail={
				'message': 'Missing required columns',
				'missing': missing,
				'required': _REQUIRED_COLS,
				'note': (
					'ต้องมี: tx_datetime, code_channel_raw, debit_amount, '
					'credit_amount, balance_amount, description_text'
				),
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


# ========= AUTO-DETECT (CSV/XLSX or PDF) → JSON =========
@router.post(
	'/predict',
	summary=(
		'Auto-detect statement file (CSV/XLSX or PDF) and analyze risk in one endpoint'
	),
	description=(
		'Upload a file:\n'
		'- CSV/XLSX (no password): table-based scoring like /predict-table.csv\n'
		'- PDF (optionally encrypted): parsed like /predict\n\n'
		'Returns JSON (PredictResponseWithDetails) with summary '
		'+ per-row transactions.\n'
		'Use query `only_fraud=true` to keep only risky rows in '
		'`transactions` (summary still from all rows).'
	),
	responses={
		200: {'description': 'OK (PredictResponseWithDetails)'},
		400: {'description': 'Invalid file'},
		403: {'description': 'Wrong password for encrypted PDF'},
		422: {'description': 'Validation / No transactions'},
		500: {'description': 'Internal error'},
	},
)
async def predict_auto(
	form: Annotated[PredictForm, Depends(PredictForm.as_form)],
	only_fraud: Annotated[
		bool,
		Query(
			description=(
				"If true, 'transactions' includes only rows where "
				'is_fraud=true (summary still computed from all rows).'
			)
		),
	] = False,
) -> PredictResponseWithDetails:
	file, password = form.file, form.password
	fname = (file.filename or '').lower()
	ext = fname.rsplit('.', 1)[-1] if '.' in fname else ''
	ctype = (file.content_type or '').lower()

	# Helper: table branch (CSV/XLSX)
	def _handle_table() -> PredictResponseWithDetails:
		# อ่านไฟล์เป็น DataFrame + normalize columns
		df = _read_table_file_to_df(file)
		df_model = _coerce_and_format(df)

		# ทำนาย
		probas = predict_proba_df(df_model)  # pd.Series index ตรงกับ df_model
		threshold = float(os.getenv('MODEL_THRESHOLD', '0.5'))
		summary = summarize_document(probas, threshold=threshold)

		# map → TxResult[]
		items: list[TxResult] = []
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

		# แยก risky
		fraud_items = [tx for tx in items if tx.is_fraud]
		fraud_indices = [tx.idx for tx in fraud_items]
		transactions_out = fraud_items if only_fraud else items

		return PredictResponseWithDetails(
			prediction=summary['prediction'],
			confidence=summary['confidence'],
			threshold=summary['threshold'],
			fraud_count=summary['fraud_count'],
			total=summary['total'],
			transactions=transactions_out,
			fraud_transactions=fraud_items,
			fraud_indices=fraud_indices,
		)

	# Helper: pdf branch
	def _handle_pdf() -> PredictResponseWithDetails:
		# ตรวจ/ปลดล็อกไฟล์ (จะ raise หากต้องใช้รหัสผ่านแต่ไม่ได้ให้มา,
		# หรือให้มาแต่ผิด)
		try:
			read_file(file.file, password)
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

		# reset pointer ก่อน extract
		file.file.seek(0)

		# Extract → df
		df, meta = extract_statement_df(file.file, password=password, return_meta=True)  # type: ignore
		if df.empty:
			detail = {
				'message': 'No transactions detected in the PDF.',
				'code': 'NO_TRANSACTIONS',
			}
			if APP_DEBUG:
				detail['debug'] = meta
			raise HTTPException(status_code=422, detail=detail)

		# validate required cols
		missing = [c for c in _REQUIRED_COLS if c not in df.columns]
		if missing:
			raise HTTPException(
				status_code=422,
				detail={
					'message': f'Missing required columns: {missing}',
					'code': 'MISSING_COLUMNS',
				},
			)

		# numeric coercion
		df_model = df[_REQUIRED_COLS].copy()
		for col in ['debit_amount', 'credit_amount', 'balance_amount']:
			df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0.0)

		# predict
		probas = predict_proba_df(df_model)
		threshold = float(os.getenv('MODEL_THRESHOLD', '0.5'))
		summary = summarize_document(probas, threshold=threshold)

		# map items
		items: list[TxResult] = []
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

		fraud_items = [tx for tx in items if tx.is_fraud]
		fraud_indices = [tx.idx for tx in fraud_items]
		transactions_out = fraud_items if only_fraud else items

		return PredictResponseWithDetails(
			prediction=summary['prediction'],
			confidence=summary['confidence'],
			threshold=summary['threshold'],
			fraud_count=summary['fraud_count'],
			total=summary['total'],
			transactions=transactions_out,
			fraud_transactions=fraud_items,
			fraud_indices=fraud_indices,
		)

	try:
		# ตัดสินใจด้วยนามสกุล/Content-Type
		is_table = ext in {'csv', 'xlsx', 'xls'} or ctype in _ALLOWED_CT
		is_pdf = ext == 'pdf' or ctype == 'application/pdf'

		if is_table and not is_pdf:
			# CSV/XLSX branch
			return _handle_table()
		elif is_pdf and not is_table:
			# PDF branch
			# สำคัญ: reset ก่อนอ่าน เพราะ FastAPI อาจอ่านไปบางส่วนแล้ว
			file.file.seek(0)
			return _handle_pdf()
		elif is_table and is_pdf:
			# แปลกมาก (แต่กันไว้)
			raise HTTPException(
				status_code=400,
				detail={
					'message': 'Ambiguous file type (both table and PDF detected).'
				},
			)
		else:
			raise HTTPException(
				status_code=400,
				detail={
					'message': 'Invalid file type. Upload a CSV/XLSX or PDF.',
					'allowed_content_types': sorted(_ALLOWED_CT | {'application/pdf'}),
				},
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

	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(
			status_code=500,
			detail={'message': 'Internal server error', 'error': str(e)},
		)
