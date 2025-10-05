# app/core/pdf_to_csv.py
from __future__ import annotations

import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import pdfplumber

# pdfminer fallback
try:
	from pdfminer.high_level import (
		extract_text as _pdfminer_extract_text,  # type: ignore
	)
except Exception:
	_pdfminer_extract_text = None

# =======================
# Config / Logging
# =======================
STRICT_AMOUNT_REQUIRED = os.getenv('STRICT_AMOUNT_REQUIRED', '0') in (
	'1',
	'true',
	'True',
)

logger = logging.getLogger('app.core.pdf_to_csv')
if not logger.handlers:
	_level = os.getenv('STATEMENT_LOG_LEVEL', 'INFO').upper()
	_handler = logging.StreamHandler()
	_handler.setFormatter(
		logging.Formatter('[%(asctime)s] [%(levelname)s] pdf_to_csv: %(message)s')
	)
	logger.addHandler(_handler)
	logger.setLevel(getattr(logging, _level, logging.INFO))


def _log_df(df: pd.DataFrame, label: str):
	try:
		logger.debug(
			f'{label}: shape={getattr(df, "shape", None)}, '
			f'cols={list(getattr(df, "columns", []))}'
		)
		if isinstance(df, pd.DataFrame) and not df.empty:
			logger.debug(f'{label} head(3):\n{df.head(3).to_string(index=False)}')
	except Exception as e:
		logger.debug(f'_log_df error: {e}')


# =======================
# Normalization helpers
# =======================
def _norm(s: str) -> str:
	s = (s or '').replace('\u200b', '').strip().lower()
	s = re.sub(r'\s+', ' ', s)
	return s


def _defkey(s: str) -> str:
	return re.sub(r'[^a-zA-Zก-ฮ0-9]+', '', _norm(s))


def _cell_norm(x):
	if x is None or (isinstance(x, float) and pd.isna(x)):
		return None
	s = str(x).replace('\u200b', '').replace('\xa0', ' ')
	s = re.sub(r'\s+', ' ', s).strip()
	return s


HEADER_MAP: Dict[str, str] = {
	# datetime
	_defkey('วันที่-เวลา'): 'tx_datetime',
	_defkey('วันเวลา'): 'tx_datetime',
	_defkey('วันที่ เวลา'): 'tx_datetime',
	_defkey('วันที่/เวลา'): 'tx_datetime',
	_defkey('date/time'): 'tx_datetime',
	_defkey('datetime'): 'tx_datetime',
	_defkey('transaction date'): 'tx_datetime',
	# channel/code
	_defkey('ช่องทาง'): 'code_channel_raw',
	_defkey('รหัสช่องทาง'): 'code_channel_raw',
	_defkey('รายการ/ช่องทาง'): 'code_channel_raw',
	_defkey('รายการ'): 'code_channel_raw',
	_defkey('code/channel'): 'code_channel_raw',
	_defkey('channel'): 'code_channel_raw',
	_defkey('code channel'): 'code_channel_raw',
	_defkey('ประเภทธุรกรรม'): 'code_channel_raw',
	_defkey('ประเภท'): 'code_channel_raw',
	# debit
	_defkey('เดบิต'): 'debit_amount',
	_defkey('ถอน'): 'debit_amount',
	_defkey('withdrawal'): 'debit_amount',
	_defkey('debit'): 'debit_amount',
	_defkey('ลูกหนี้'): 'debit_amount',
	# credit
	_defkey('เครดิต'): 'credit_amount',
	_defkey('ฝาก'): 'credit_amount',
	_defkey('deposit'): 'credit_amount',
	_defkey('credit'): 'credit_amount',
	_defkey('เจ้าหนี้'): 'credit_amount',
	# balance
	_defkey('คงเหลือ'): 'balance_amount',
	_defkey('ยอดคงเหลือ'): 'balance_amount',
	_defkey('balance'): 'balance_amount',
	_defkey('available balance'): 'balance_amount',
	# description
	_defkey('รายละเอียด'): 'description_text',
	_defkey('บันทึกช่วยจำ'): 'description_text',
	_defkey('description/note'): 'description_text',
	_defkey('description'): 'description_text',
	_defkey('note'): 'description_text',
}
# bilingual header merged
HEADER_MAP.update(
	{
		_defkey('date/time วันที่/เวลา'): 'tx_datetime',
		_defkey('code/channel รายการ/ช่องทาง'): 'code_channel_raw',
		_defkey('debit ลูกหนี้'): 'debit_amount',
		_defkey('credit เจ้าหนี้'): 'credit_amount',
		_defkey('balance ยอดคงเหลือ'): 'balance_amount',
		_defkey('description/note รายละเอียด/บันทึกช่วยจำ'): 'description_text',
	}
)

TARGET_COLS = [
	'tx_datetime',
	'code_channel_raw',
	'debit_amount',
	'credit_amount',
	'balance_amount',
	'description_text',
]

# =======================
# Heuristics
# =======================
_DT_PATTERNS = [
	re.compile(r'^\d{4}-\d{2}-\d{2}[ T]\d{1,2}:\d{2}$'),
	re.compile(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+\d{1,2}:\d{2}$'),
	re.compile(r'^\d{1,2}\s+[A-Za-z]{3}\s+\d{4}\s+\d{1,2}:\d{2}$'),
]
_HEADER_NOISE_TERMS = [
	'account statement',
	'statement period',
	'page',
	'account no',
	'account type',
	'รายการเดินบัญชี',
	'บันทึกช่วยจำ',
	'เลขที่บัญชี',
	'ประเภทบัญชี',
	'วันที่',
	'หน้า',
	'savings',
	'current',
	'พร้อมบันทึกช่วยจํา',
	'รายการเดินบัญชี พร้อมบันทึกช่วยจํา',
]
_HEADER_NOISE_RE = re.compile(
	'|'.join([re.escape(x) for x in _HEADER_NOISE_TERMS]), re.IGNORECASE
)


def _looks_tx_datetime(s: Optional[str]) -> bool:
	if not s:
		return False
	s = re.sub(r'\s+', ' ', str(s).strip())
	for pat in _DT_PATTERNS:
		if pat.match(s):
			return True
	if re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', s):  # date only
		return True
	return False


def _is_tx_row(row: pd.Series) -> bool:
	dt = row.get('tx_datetime')
	desc = row.get('description_text') or ''
	chan = row.get('code_channel_raw') or ''
	joined = f'{(dt or "")} {(chan or "")} {(desc or "")}'.strip()

	if _HEADER_NOISE_RE.search(joined):
		return False

	if not _looks_tx_datetime(str(dt) if dt is not None else None):
		return False

	if STRICT_AMOUNT_REQUIRED:
		has_amount = any(
			x is not None
			for x in (
				row.get('debit_amount'),
				row.get('credit_amount'),
				row.get('balance_amount'),
			)
		)
		if not has_amount:
			return False
		return True
	else:
		# ผ่อน: มี dt + (amount หรือ code หรือ desc อย่างน้อยหนึ่ง)
		has_any = any(
			[
				any(
					x is not None
					for x in (
						row.get('debit_amount'),
						row.get('credit_amount'),
						row.get('balance_amount'),
					)
				),
				bool((chan or '').strip()),
				bool((desc or '').strip()),
			]
		)
		return has_any


_amount_clean = re.compile(r'[^0-9\.\-]+')


def _to_float_or_none(x):
	if x is None:
		return None
	s = str(x).strip()
	if s == '' or s.lower() in {'-', 'na', 'none'}:
		return None
	neg = s.startswith('(') and s.endswith(')')
	s = s.replace('(', '').replace(')', '')
	s = s.replace('฿', '').replace(',', '').replace(' ', '')
	s = _amount_clean.sub('', s)
	if s in {'', '-', '.', '-.'}:
		return None
	try:
		v = float(s)
		return -v if neg else v
	except Exception:
		return None


# ---------- Thai/EN date helpers ----------
TH_MONTH = {
	'ม.ค.': 'Jan',
	'ก.พ.': 'Feb',
	'มี.ค.': 'Mar',
	'เม.ย.': 'Apr',
	'พ.ค.': 'May',
	'มิ.ย.': 'Jun',
	'ก.ค.': 'Jul',
	'ส.ค.': 'Aug',
	'ก.ย.': 'Sep',
	'ต.ค.': 'Oct',
	'พ.ย.': 'Nov',
	'ธ.ค.': 'Dec',
	'ม.ค': 'Jan',
	'ก.พ': 'Feb',
	'มี.ค': 'Mar',
	'เม.ย': 'Apr',
	'พ.ค': 'May',
	'มิ.ย': 'Jun',
	'ก.ค': 'Jul',
	'ส.ค': 'Aug',
	'ก.ย': 'Sep',
	'ต.ค': 'Oct',
	'พ.ย': 'Nov',
	'ธ.ค': 'Dec',
}


def _thai_date_to_iso(dt_str: str) -> str:
	import datetime as dt

	s = (dt_str or '').strip()
	m = re.search(r'(\d{1,2})\s+([ก-ฮ\.]+)\s+(\d{4})(?:\s+(\d{1,2}:\d{2}))?', s)
	if m:
		day, th_mon, year, hm = m.groups()
		year = int(year) - 543
		mon_en = TH_MONTH.get(th_mon.strip(), None)
		if mon_en:
			if not hm:
				hm = '00:00'
			try:
				d = dt.datetime.strptime(
					f'{day} {mon_en} {year} {hm}', '%d %b %Y %H:%M'
				)
				return d.strftime('%Y-%m-%d %H:%M')
			except Exception:
				pass
	m2 = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{2,4})\s+(\d{1,2}):(\d{2})$', s)
	if m2:
		d, m, y, hh, mm = m2.groups()
		y = int(y)
		if y < 100:
			y += 2000 if y < 70 else 1900
		return f'{y:04d}-{int(m):02d}-{int(d):02d} {int(hh):02d}:{int(mm):02d}'
	return s


# ---------- header mapping (fuzzy) ----------
def _map_header_name(s: str) -> Optional[str]:
	key = HEADER_MAP.get(_defkey(s))
	if key:
		return key
	s2 = _norm(s)
	if any(k in s2 for k in ['date', 'เวลา', 'วันที่', 'datetime']):
		return 'tx_datetime'
	if any(k in s2 for k in ['code', 'channel', 'ช่องทาง', 'รายการ', 'ประเภทธุรกรรม']):
		return 'code_channel_raw'
	if any(k in s2 for k in ['debit', 'เดบิต', 'ถอน', 'ลูกหนี้']):
		return 'debit_amount'
	if any(k in s2 for k in ['credit', 'เครดิต', 'ฝาก', 'เจ้าหนี้']):
		return 'credit_amount'
	if any(k in s2 for k in ['balance', 'คงเหลือ', 'ยอดคงเหลือ']):
		return 'balance_amount'
	if any(k in s2 for k in ['description', 'note', 'รายละเอียด', 'บันทึก']):
		return 'description_text'
	return None


def _apply_header_map(cols: List[str]) -> List[str]:
	mapped = []
	for c in cols:
		c = str(c or '').replace('\n', ' ').strip()
		key = _map_header_name(c)
		mapped.append(key if key else _norm(c))
	return mapped


def _strip_or_none(x):
	return x.strip() if isinstance(x, str) else None


# ---------- merge duplicate mapped columns ----------
_date_only_re = re.compile(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$')
_time_only_re = re.compile(r'^\d{1,2}:\d{2}$')


def _drop_columns_safe(df: pd.DataFrame, cols_to_drop: List[str]) -> pd.DataFrame:
	safe = [
		c
		for c in dict.fromkeys(cols_to_drop)
		if isinstance(c, str) and c != '' and c in df.columns
	]
	if not safe:
		return df
	try:
		return df.drop(columns=safe, errors='ignore')
	except Exception as e:
		logger.debug(f'_drop_columns_safe fallback: {e} | trying column selection')
		keep = [c for c in df.columns if c not in safe]
		return df.loc[:, keep]


def _merge_duplicate_mapped_cols(df: pd.DataFrame) -> pd.DataFrame:
	cols = list(df.columns)
	dup_groups: Dict[str, List[int]] = {}
	for i, c in enumerate(cols):
		dup_groups.setdefault(c, []).append(i)

	for col, idxs in dup_groups.items():
		if len(idxs) <= 1:
			continue
		base = cols[idxs[0]]
		for j in idxs[1:]:
			other = df.iloc[:, j]
			if col == 'tx_datetime':

				def _join_dt(a, b):
					a = (None if pd.isna(a) else str(a).strip()) or None
					b = (None if pd.isna(b) else str(b).strip()) or None
					if _date_only_re.match(a or '') and _time_only_re.match(b or ''):
						return f'{a} {b}'
					if _date_only_re.match(b or '') and _time_only_re.match(a or ''):
						return f'{b} {a}'
					return a or b

				df[base] = [_join_dt(x, y) for x, y in zip(df[base], other)]
			else:
				df[base] = df[base].combine_first(other)
		drop_cols = [cols[i] for i in idxs[1:]]
		df = _drop_columns_safe(df, drop_cols)
		cols = list(df.columns)
	return df


def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
	t0 = time.time()
	df = df.copy()
	# pandas>=2.2: ใช้ map แทน applymap
	df = df.map(_cell_norm)
	_log_df(df, 'raw-in')

	df.columns = _apply_header_map([str(c) for c in df.columns])
	df = _merge_duplicate_mapped_cols(df)
	_log_df(df, 'after-merge-dup')

	for col in TARGET_COLS:
		if col not in df.columns:
			df[col] = None
	df = df[TARGET_COLS]
	_log_df(df, 'after-reindex-targets')

	df['tx_datetime'] = df['tx_datetime'].map(_cell_norm)

	for c in ('debit_amount', 'credit_amount', 'balance_amount'):
		df[c] = df[c].map(_to_float_or_none)
	_log_df(df, 'after-amount-cast')

	for tcol in ('tx_datetime', 'code_channel_raw', 'description_text'):
		df[tcol] = df[tcol].apply(_strip_or_none)

	df['tx_datetime'] = df['tx_datetime'].apply(
		lambda s: _thai_date_to_iso(s) if isinstance(s, str) else s
	)
	_log_df(df, 'after-date-iso')

	desc = df['description_text'].fillna('')
	before = len(df)
	df = df[~desc.str.contains('balance brought forward', case=False)]
	df = df[~desc.str.contains('ยอดคง.*ยกมา', case=False)]
	logger.debug(f'filter brought-forward: {before} -> {len(df)}')

	if not df.empty:
		before = len(df)
		df = df[df.apply(_is_tx_row, axis=1)]
		logger.debug(f'filter tx rows: {before} -> {len(df)}')
	_log_df(df, 'after-filter-tx-rows')

	before = len(df)
	df = df[~(df['tx_datetime'].isna() & df['description_text'].isna())].reset_index(
		drop=True
	)
	logger.debug(f'filter blank rows: {before} -> {len(df)}')

	if not df.empty:
		before = len(df)
		df = df.drop_duplicates(subset=TARGET_COLS, keep='first').reset_index(drop=True)
		logger.debug(f'drop_duplicates: {before} -> {len(df)}')

	logger.debug(f'_standardize_df took {time.time() - t0:.3f}s')
	return df


# ---------- table extraction ----------
TABLE_SETTING_CANDIDATES = [
	dict(
		vertical_strategy='lines',
		horizontal_strategy='lines',
		text_x_tolerance=2,
		text_y_tolerance=2,
		intersection_x_tolerance=5,
		intersection_y_tolerance=5,
		snap_x_tolerance=2,
		snap_y_tolerance=2,
		join_tolerance=2,
		edge_min_length=40,
	),
	dict(
		vertical_strategy='text',
		horizontal_strategy='text',
		text_x_tolerance=2,
		text_y_tolerance=2,
	),
	dict(
		vertical_strategy='lines',
		horizontal_strategy='text',
		text_x_tolerance=2,
		text_y_tolerance=2,
	),
	dict(
		vertical_strategy='text',
		horizontal_strategy='lines',
		text_x_tolerance=2,
		text_y_tolerance=2,
	),
]


def _looks_thai_row(row: List[str]) -> bool:
	s = ' '.join([str(x or '') for x in row])
	return bool(re.search(r'[ก-ฮ]', s))


_TH_HEADER_HINTS = {
	'วันที่',
	'เวลา',
	'รายการ',
	'ช่องทาง',
	'ลูกหนี้',
	'เจ้าหนี้',
	'คงเหลือ',
	'ยอดคงเหลือ',
	'รายละเอียด',
	'บันทึก',
}


def _maybe_bilingual_header(h1: List[str], h2: Optional[List[str]]) -> List[str]:
	if not h2:
		return [str(x or '').strip() for x in h1]
	hint = ' '.join([str(x or '') for x in h2])
	if any(k in hint for k in _TH_HEADER_HINTS):
		out = []
		for i in range(max(len(h1), len(h2))):
			a = str(h1[i]) if i < len(h1) and h1[i] is not None else ''
			b = str(h2[i]) if i < len(h2) and h2[i] is not None else ''
			out.append(' '.join([a.strip(), b.strip()]).strip())
		return out
	return [str(x or '').strip() for x in h1]


def _extract_tables_with_pdfplumber(
	file_obj, password: Optional[str]
) -> List[pd.DataFrame]:
	frames: List[pd.DataFrame] = []
	with pdfplumber.open(file_obj, password=password) as pdf:
		for page in pdf.pages:
			tables_all: List[List[List[str]]] = []
			for ts in TABLE_SETTING_CANDIDATES:
				try:
					tables = page.extract_tables(table_settings=ts) or []
					tables_all.extend(tables)
				except Exception:
					continue
			for tbl in tables_all:
				if not tbl or not any(tbl):
					continue
				header_idx = 0
				for i, row in enumerate(tbl[:5]):
					if any((str(x or '').strip() for x in row)):
						header_idx = i
						break
				header_raw = [str(x or '').strip() for x in tbl[header_idx]]
				next_row = tbl[header_idx + 1] if header_idx + 1 < len(tbl) else None
				header = _maybe_bilingual_header(header_raw, next_row)
				body_start = header_idx + (
					2
					if next_row is not None
					and any(k in ' '.join(map(str, next_row)) for k in _TH_HEADER_HINTS)
					else 1
				)
				body = [
					r
					for r in tbl[body_start:]
					if any((str(x or '').strip() for x in r))
				]
				if not body:
					continue
				try:
					df = pd.DataFrame(body, columns=header)
					df = _standardize_df(df)
					if not df.empty:
						frames.append(df)
				except Exception:
					pass
	return frames


# ---------- words-projection with SCB hard bounds fallback ----------
HEADER_KEYS = [
	(
		'tx_datetime',
		['date', 'date/time', 'datetime', 'วันที่', 'เวลา', 'วันที่/เวลา', 'วัน/เวลา'],
	),
	(
		'code_channel_raw',
		[
			'code',
			'channel',
			'code/channel',
			'รายการ',
			'ช่องทาง',
			'รายการ/ช่องทาง',
			'ประเภทธุรกรรม',
		],
	),
	('debit_amount', ['debit', 'เดบิต', 'ลูกหนี้', 'ถอน', 'withdrawal']),
	('credit_amount', ['credit', 'เครดิต', 'เจ้าหนี้', 'ฝาก', 'deposit']),
	('balance_amount', ['balance', 'ยอดคงเหลือ', 'คงเหลือ']),
	(
		'description_text',
		['description', 'note', 'description/note', 'รายละเอียด', 'บันทึก', 'บันทึกช่วยจำ'],
	),
]

SCB_RELATIVE_BOUNDS = [  # ซ้าย->ขวา (สัดส่วนความกว้างหน้า)
	('tx_datetime', 0.00, 0.17),
	('code_channel_raw', 0.17, 0.36),
	('debit_amount', 0.36, 0.48),
	('credit_amount', 0.48, 0.60),
	('balance_amount', 0.60, 0.74),
	('description_text', 0.74, 1.00),
]


def _categorize_header_token(txt: str) -> Optional[str]:
	t = _norm(txt)
	for key, variants in HEADER_KEYS:
		for v in variants:
			if _norm(v) in t:
				return key
	return None


def _infer_header_bounds_from_top(page) -> List[Tuple[str, float, float]]:
	words = page.extract_words(extra_attrs=['x0', 'x1', 'top', 'bottom']) or []
	if not words:
		return []
	H = page.height
	top_words = [w for w in words if w['top'] < H * 0.40]  # ขยายเป็น 40%
	buckets: Dict[str, List[float]] = {}
	for w in top_words:
		key = _categorize_header_token(w['text'])
		if key:
			mid = (w['x0'] + w['x1']) / 2.0
			buckets.setdefault(key, []).append(mid)
	if len(buckets) < 3:
		return []
	centers = {k: sum(v) / len(v) for k, v in buckets.items()}
	ordered = sorted(centers.items(), key=lambda kv: kv[1])
	bounds: List[Tuple[str, float, float]] = []
	for i, (k, mid) in enumerate(ordered):
		left_mid = ordered[i - 1][1] if i > 0 else 0.0
		right_mid = ordered[i + 1][1] if i + 1 < len(ordered) else page.width
		x0 = (left_mid + mid) / 2 if i > 0 else 0.0
		x1 = (mid + right_mid) / 2 if i + 1 < len(ordered) else page.width
		bounds.append((k, x0, x1))
	return bounds


def _assign_cells_from_tokens(
	tokens: List[dict], bounds: List[Tuple[str, float, float]]
) -> Dict[str, str]:
	cells: Dict[str, str] = {k: '' for k, *_ in bounds}
	for tok in tokens:
		mid = (tok['x0'] + tok['x1']) / 2.0
		for k, x0, x1 in bounds:
			if x0 - 1 <= mid <= x1 + 1:
				cells[k] = (cells[k] + ' ' + tok['text']).strip()
				break
	return cells


def _extract_by_words_projection(file_obj, password: Optional[str]) -> pd.DataFrame:
	rows_all: List[Dict] = []
	with pdfplumber.open(file_obj, password=password) as pdf:
		for page in pdf.pages:
			bounds = _infer_header_bounds_from_top(page)
			if not bounds:
				# ใช้ขอบเขตคอลัมน์แบบตายตัวของ SCB (สัดส่วน)
				W = page.width
				bounds = [(k, W * a, W * b) for (k, a, b) in SCB_RELATIVE_BOUNDS]

			words = page.extract_words(extra_attrs=['x0', 'x1', 'top', 'bottom']) or []
			if not words:
				continue

			# หา y เริ่มข้อมูล: จากบรรทัดแรกที่มี pattern วันที่
			start_y = (
				min(
					[
						w['top']
						for w in words
						if re.search(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', w['text'])
					],
					default=page.height * 0.25,
				)
				- 4
			)

			# cluster tokens เป็นบรรทัดเฉพาะส่วนข้อมูล
			y_tol = 3.0
			lines: List[List[dict]] = []
			for w in sorted(
				(t for t in words if t['top'] >= start_y),
				key=lambda d: (d['top'], d['x0']),
			):
				if not lines or abs(lines[-1][0]['top'] - w['top']) > y_tol:
					lines.append([w])
				else:
					lines[-1].append(w)

			i = 0
			while i < len(lines):
				line_tokens = lines[i]
				line_text = ' '.join(t['text'] for t in line_tokens).strip()
				if _HEADER_NOISE_RE.search(line_text.lower()):
					i += 1
					continue

				has_date_hint = bool(
					re.search(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', line_text)
					or re.search(r'\b\d{4}-\d{2}-\d{2}\b', line_text)
				)
				if not has_date_hint:
					i += 1
					continue

				base = _assign_cells_from_tokens(line_tokens, bounds)

				# รวมบรรทัดถัดไปจนกว่าจะเจอวันที่ใหม่
				j = i + 1
				while j < len(lines):
					nxt_tokens = lines[j]
					nxt_text = ' '.join(t['text'] for t in nxt_tokens).strip()
					if re.search(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', nxt_text) or re.search(
						r'\b\d{4}-\d{2}-\d{2}\b', nxt_text
					):
						break
					nxt_cells = _assign_cells_from_tokens(nxt_tokens, bounds)
					# ต่อเวลา
					if base.get('tx_datetime') and _time_only_re.match(
						(nxt_cells.get('tx_datetime') or '').strip()
					):
						base['tx_datetime'] = (
							base['tx_datetime'] + ' ' + nxt_cells['tx_datetime']
						).strip()
					# เติมช่องว่างที่ยังไม่มี
					for k in [
						'code_channel_raw',
						'debit_amount',
						'credit_amount',
						'balance_amount',
					]:
						if (not base.get(k)) and nxt_cells.get(k):
							base[k] = nxt_cells[k]
					# ต่อ description / DESC / NOTE
					dtxt = nxt_cells.get('description_text', '')
					if dtxt:
						base['description_text'] = (
							base.get('description_text', '')
							+ ('\n' if base.get('description_text') else '')
							+ dtxt
						).strip()
					j += 1

				rows_all.append(
					{
						'tx_datetime': base.get('tx_datetime') or None,
						'code_channel_raw': base.get('code_channel_raw') or None,
						'debit_amount': base.get('debit_amount') or None,
						'credit_amount': base.get('credit_amount') or None,
						'balance_amount': base.get('balance_amount') or None,
						'description_text': base.get('description_text') or None,
					}
				)
				i = j

	if not rows_all:
		return pd.DataFrame(columns=TARGET_COLS)
	df = pd.DataFrame(rows_all, columns=TARGET_COLS)
	_log_df(df, 'words-projection raw')
	return _standardize_df(df)


# ---------- text fallback (gaps; keep as last resort) ----------
def _extract_lines_with_pdfminer_or_plumber(
	file_obj, password: Optional[str]
) -> List[str]:
	lines: List[str] = []
	if _pdfminer_extract_text is not None:
		try:
			file_obj.seek(0)
			txt = _pdfminer_extract_text(file_obj) or ''
			tmp = [ln.strip() for ln in txt.splitlines() if ln.strip()]
			if tmp:
				lines = tmp
				file_obj.seek(0)
				return lines
		except Exception:
			pass

	file_obj.seek(0)
	with pdfplumber.open(file_obj, password=password) as pdf:
		for p in pdf.pages:
			text = p.extract_text() or ''
			for ln in text.splitlines():
				ln = ln.strip()
				if ln:
					lines.append(ln)
	file_obj.seek(0)
	return lines


def _parse_text_rows_by_gaps(lines: List[str]) -> pd.DataFrame:
	rows: List[Dict] = []
	pending: Optional[Dict] = None
	date_pat = re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b')
	time_pat = re.compile(r'\b\d{1,2}:\d{2}\b')

	for ln in lines:
		if not ln:
			continue
		if date_pat.search(ln):
			if pending:
				rows.append(pending)
			pending = {
				'tx_datetime': None,
				'code_channel_raw': None,
				'debit_amount': None,
				'credit_amount': None,
				'balance_amount': None,
				'description_text': None,
			}
			# date + time ในบรรทัดเดียว
			dt = ln.strip()
			if time_pat.search(dt) and '/' in dt:
				# เก็บทั้งบรรทัดเป็น dt_raw แล้วให้ standardizer แปลง
				pending['tx_datetime'] = (
					re.search(r'\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}', dt).group(0)
					if re.search(r'\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}', dt)
					else dt
				)
			else:
				# เก็บเฉพาะวันที่ไว้ก่อน รอเวลาในบรรทัดถัดไป
				date_only = re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', dt).group(0)
				pending['tx_datetime'] = date_only

			continue

		if pending:
			# ต่อเวลา
			if pending.get('tx_datetime') and (
				time_pat.search(ln) and ':' in ln and '/' not in ln
			):
				pending['tx_datetime'] = (
					f'{pending["tx_datetime"]} {time_pat.search(ln).group(0)}'
				)
				continue

			# map คำหลักแบบหยาบ (จะถูกมาตรฐานใน _standardize_df อีกครั้ง)
			if any(
				k in _norm(ln)
				for k in [
					'x1',
					'x2',
					'enet',
					'pos',
					'sipi',
					'atm',
					'trf',
					'kb',
					'ktb',
					'scb',
				]
			):
				pending['code_channel_raw'] = (
					(pending.get('code_channel_raw') or '')
					+ (' ' if pending.get('code_channel_raw') else '')
					+ ln
				)

			# เดาเลขจำนวนเงินในบรรทัด (จับตัวแรก ๆ)
			money = re.findall(r'[()\-]?\d[\d,]*(?:\.\d{1,2})?', ln)
			if money:
				# ใส่ให้ช่องที่ยังว่างตามลำดับ debit -> credit -> balance
				for field in ('debit_amount', 'credit_amount', 'balance_amount'):
					if pending.get(field) is None:
						pending[field] = money[0]
						money = money[1:]
					if not money:
						break

			# ต่อ description (รวม DESC/NOTE)
			if pending.get('description_text'):
				pending['description_text'] = (
					pending['description_text'] + ' ' + ln
				).strip()
			else:
				pending['description_text'] = ln

	if pending:
		rows.append(pending)

	if not rows:
		return pd.DataFrame(columns=TARGET_COLS)
	df = pd.DataFrame(rows, columns=TARGET_COLS)
	return _standardize_df(df)


# ---------- public API ----------
def extract_statement_df(
	file_obj, password: Optional[str] = None, return_meta: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
	meta: Dict[str, Dict] = {'tables': {}, 'words': {}, 'fallback': {}}

	# 1) tables-first
	file_obj.seek(0)
	frames = _extract_tables_with_pdfplumber(file_obj, password=password)
	meta['tables']['frames'] = len(frames)
	meta['tables']['rows_total'] = sum(len(f) for f in frames)
	if frames:
		out = pd.concat(
			[f for f in frames if isinstance(f, pd.DataFrame)], ignore_index=True
		)
		meta['tables']['rows_after_concat'] = len(out)
		if not out.empty:
			return (out, meta) if return_meta else out

	# 2) words-projection (robust + SCB hard bounds)
	file_obj.seek(0)
	df_words = _extract_by_words_projection(file_obj, password=password)
	meta['words']['rows'] = len(df_words)
	if not df_words.empty:
		return (df_words, meta) if return_meta else df_words

	# 3) text fallback (gaps + greedy)
	lines = _extract_lines_with_pdfminer_or_plumber(file_obj, password=password)
	meta['fallback']['lines'] = len(lines)
	df = _parse_text_rows_by_gaps(lines)
	meta['fallback']['rows'] = len(df)

	return (df, meta) if return_meta else df


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
	return df.to_csv(index=False).encode('utf-8')
