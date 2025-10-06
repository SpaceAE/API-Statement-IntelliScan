# app/api/routes/statements.py
import io
import os
import re
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.file import (
	IncorrectPasswordException,
	PasswordRequiredException,
	read_file,  # ต้องอัปเดตให้คืน bytes และตรวจ PDF ที่เข้ารหัสใน file.py
)
from app.core.model import predict_proba_df, summarize_document
from app.core.pdfToCSV import (  # ✅ ใช้ชื่อไฟล์ใหม่ pdfToCSV.py
	StatementFormatUnsupported,
	StatementParsingFailed,
	extract_statement_df,
)
from app.models.request import PredictForm
from app.models.response import (
	PredictResponse,
	TransactionResult,
)

router = APIRouter(prefix='/statements', tags=['Statements'])

# ===== Required columns (ใช้ร่วมทั้ง CSV/XLSX และผลจาก PDF Extract) =====
_REQUIRED_COLS = [
	'tx_datetime',
	'code_channel_raw',
	'debit_amount',
	'credit_amount',
	'balance_amount',
	'description_text',
]

# Content-Type ที่อนุญาตสำหรับ table
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
			df = pd.read_excel(bio, engine='openpyxl')
		else:
			# fallback ลอง CSV
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

	# amounts
	for col in ['debit_amount', 'credit_amount', 'balance_amount']:
		out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0.0)

	# datetime -> DD/MM/YYYY HH:mm (ฝั่งโมเดลของคุณอาจไม่บังคับฟอร์แมตนี้ แต่คงไว้ตามเดิม)
	dt_parsed = pd.to_datetime(out['tx_datetime'], errors='coerce', dayfirst=True)
	mask = dt_parsed.notna()
	out.loc[mask, 'tx_datetime'] = dt_parsed[mask].dt.strftime('%d/%m/%Y %H:%M')

	out['description_text'] = out['description_text'].astype(str).fillna('')
	return out


@router.post(
	'/predict',
	summary='Analyze Statement Risk (CSV/XLSX or PDF)',
	description=(
		'Upload a statement file (CSV/XLSX or PDF; PDF may be password-protected). '
		'Auto-detects type, parses transactions, scores risk per row and returns JSON\n'
		'Use query `only_fraud=true` to include only risky rows in `transactions`.'
	),
	responses={
		200: {'description': 'OK (PredictResponse)'},
		400: {'description': 'Invalid file'},
		403: {'description': 'Wrong password for encrypted PDF'},
		422: {'description': 'Validation / No transactions'},
		500: {'description': 'Internal error'},
	},
)
async def predict(
	form: Annotated[PredictForm, Depends(PredictForm.as_form)],
	only_fraud: Annotated[
		bool,
		Query(
			description=(
				"If true, 'transactions' includes only rows where is_fraud=true "
				'(summary still computed from all rows).'
			)
		),
	] = False,
) -> PredictResponse:
	"""
	Flow:
	- ใช้ file.py: read_file(…) -> bytes + ตรวจ PDF (รหัสผ่าน/เข้ารหัส)
	- ถ้าเป็น PDF -> extract_statement_df(BytesIO, password)
	- ถ้าเป็น CSV/XLSX -> pandas.read_* แล้ว normalize
	- ส่งเข้าโมเดล -> summarize -> สร้างรายการ TransactionResult[]
	"""
	up = form.file
	password = form.password
	fname = (up.filename or '').lower()
	ext = fname.rsplit('.', 1)[-1] if '.' in fname else ''
	ctype = (up.content_type or '').lower()

	# 1) อ่านทั้งหมดเป็น bytes + ให้ file.py ตรวจ PDF encryption/password
	try:
		raw = read_file(up.file, password)  # ✅ ควรอัปเดต read_file ให้คืน bytes เสมอ
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

	# 2) ตัดสินชนิดไฟล์จากนามสกุล/Content-Type
	is_pdf = ext == 'pdf' or ctype == 'application/pdf'
	is_table = (ext in {'csv', 'xlsx', 'xls'}) or (ctype in _ALLOWED_CT)

	# 3) แตกแขนง
	try:
		if is_pdf and not is_table:
			# ----- PDF branch -----
			bio = io.BytesIO(raw)
			df, meta = extract_statement_df(bio, password=password, return_meta=True)  # type: ignore
			if df.empty:
				raise HTTPException(
					status_code=422,
					detail={'message': 'No transactions detected in the PDF.'},
				)
		elif is_table and not is_pdf:
			# ----- CSV/XLSX branch -----
			df = _read_tb_file_to_df(fname, ctype, raw)
		elif is_table and is_pdf:
			# กรณีแปลก
			raise HTTPException(
				status_code=400,
				detail={'message': 'Ambiguous file type (both table & PDF detected).'},
			)
		else:
			# ไม่เข้าเงื่อนไข
			raise HTTPException(
				status_code=400,
				detail={
					'message': 'Invalid file type. Upload a CSV/XLSX or PDF.',
					'allowed_content_types': sorted(_ALLOWED_CT | {'application/pdf'}),
				},
			)
	except StatementFormatUnsupported as e:
		raise HTTPException(
			status_code=422, detail={'message': str(e), 'code': 'UNSUPPORTED_BANK'}
		)
	except StatementParsingFailed as e:
		raise HTTPException(
			status_code=422, detail={'message': str(e), 'code': 'PARSE_FAILED'}
		)

	# 4) validate required cols + cast/format
	missing = [c for c in _REQUIRED_COLS if c not in df.columns]
	if missing:
		raise HTTPException(
			status_code=422,
			detail={'message': f'Missing required columns: {missing}'},
		)

	df_model = _coerce_and_format(df)

	# 5) predict per row
	probas = predict_proba_df(df_model)  # pd.Series
	threshold = float(os.getenv('MODEL_THRESHOLD', '0.5'))
	summary = summarize_document(probas, threshold=threshold)

	# 6) build items
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

	# หลังจาก build items และสรุป summary แล้ว
	fraud_items = [tx for tx in items if tx.is_fraud]
	fraud_count = len(fraud_items)
	transactions_total = len(df_model)  # จำนวนรายการที่ประมวลผลทั้งหมด

	transactions_out = fraud_items if only_fraud else items

	return PredictResponse(
		prediction=summary['prediction'],
		confidence=summary['confidence'],
		fraud_count=fraud_count,
		transactions_count=transactions_total,
		transactions=transactions_out,
	)
