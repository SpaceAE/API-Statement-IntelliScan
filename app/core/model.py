from __future__ import annotations

import json
import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from keras.models import load_model

from app.core.config import settings

# -----------------------------
# Global caches
# -----------------------------
_cached_model = None
_cached_tfidf = None
_cached_scaler = None
_cached_vocab = None  # {"code_channel_raw": ["X1/ENET", ...]}
_cached_numlist = None  # numeric feature list (fallback)
_cached_meta = None  # model_meta.json

# ใช้โฟลเดอร์เดียวกับโมเดลเสมอ
ARTIFACT_DIR = os.path.dirname(settings.MODEL_PATH)


def _p(name: str) -> str:
	"""Join path with artifact dir."""
	return os.path.join(ARTIFACT_DIR, name)


# -----------------------------
# Model / Artifacts loaders
# -----------------------------
def get_model():
	"""
	โหลด Keras model (.h5) จาก settings.MODEL_PATH (compile=False)
	"""
	global _cached_model
	if _cached_model is None:
		if not os.path.exists(settings.MODEL_PATH):
			raise FileNotFoundError(f'Model file not found at {settings.MODEL_PATH}')
		_cached_model = load_model(settings.MODEL_PATH, compile=False)
	return _cached_model


def _load_artifacts():
	"""
	โหลดตัวแปลง/เมตาให้ครบจาก ARTIFACT_DIR:
	- pre_tfidf.joblib
	- pre_scaler.joblib
	- pre_categ_vocab.json (fallback vocab)
	- pre_numeric_features.json (fallback numeric list)
	- model_meta.json (tfidf_dim, vocab, numeric_features, threshold, total_input_dim)
	"""
	global _cached_tfidf, _cached_scaler, _cached_vocab, _cached_numlist, _cached_meta

	if _cached_tfidf is None:
		_cached_tfidf = joblib.load(_p('pre_tfidf.joblib'))

	if _cached_scaler is None:
		_cached_scaler = joblib.load(_p('pre_scaler.joblib'))

	if _cached_vocab is None and os.path.exists(_p('pre_categ_vocab.json')):
		with open(_p('pre_categ_vocab.json'), 'r', encoding='utf-8') as f:
			_cached_vocab = json.load(f)

	if _cached_numlist is None and os.path.exists(_p('pre_numeric_features.json')):
		with open(_p('pre_numeric_features.json'), 'r', encoding='utf-8') as f:
			_cached_numlist = json.load(f)

	if _cached_meta is None and os.path.exists(_p('model_meta.json')):
		with open(_p('model_meta.json'), 'r', encoding='utf-8') as f:
			_cached_meta = json.load(f)


def _numeric_feature_names_from_scaler() -> Optional[List[str]]:
	"""
	เลือกลำดับฟีเจอร์เชิงตัวเลขตามลำดับ:
	1) model_meta.json['numeric_features']
	2) scaler.feature_names_in_ (ถ้ามี)
	3) pre_numeric_features.json (สำรอง)
	"""
	if _cached_meta and isinstance(_cached_meta, dict):
		meta_nums = _cached_meta.get('numeric_features')
		if isinstance(meta_nums, list) and meta_nums:
			return meta_nums

	names = getattr(_cached_scaler, 'feature_names_in_', None)
	if names is not None:
		return list(names)

	return _cached_numlist


# -----------------------------
# Feature engineering
# -----------------------------
def _coerce_base(df: pd.DataFrame) -> pd.DataFrame:
	"""
	ทำความสะอาด/จัดรูปอินพุตพื้นฐานจาก df อินพุตของผู้ใช้
	- รองรับ tx_datetime รูปแบบ '%d/%m/%Y %H:%M'
	- upper() code_channel_raw (ควรให้ฝั่งเทรนใช้กฎเดียวกัน)
	"""
	base = pd.DataFrame(index=df.index)

	for c in ['debit_amount', 'credit_amount', 'balance_amount']:
		base[c] = pd.to_numeric(df.get(c), errors='coerce').fillna(0.0)

	dt = pd.to_datetime(
		df.get('tx_datetime'),
		format='%d/%m/%Y %H:%M',
		errors='coerce',
	)
	base['__dt__'] = dt

	# แยกตัวแปรกลาง เลี่ยงบรรทัดยาวและหลีกเลี่ยงการปน tabs/spaces
	ccr = df.get('code_channel_raw')
	base['code_channel_raw'] = ccr.astype(str).fillna('').str.upper()

	desc = df.get('description_text')
	base['description_text'] = desc.astype(str).fillna('')

	return base


def _build_numeric_features(df_base: pd.DataFrame) -> pd.DataFrame:
	"""
	สร้างฟีเจอร์เชิงตัวเลขพื้นฐาน (12 ฟีเจอร์)
	แล้วค่อย reindex ตามลิสต์ที่ meta/scaler แจ้ง
	"""
	debit = df_base['debit_amount'].astype(float)
	credit = df_base['credit_amount'].astype(float)
	balance = df_base['balance_amount'].astype(float)
	dt = df_base['__dt__']

	# ฟีเจอร์พื้นฐาน
	net_amount = credit - debit
	abs_debit = debit.abs()
	abs_credit = credit.abs()
	total_flow = abs_debit + abs_credit

	# สถานะประเภทบรรทัด
	is_debit = (debit > 0).astype(float)
	is_credit = (credit > 0).astype(float)

	# เวลา
	hour = dt.dt.hour.fillna(0).astype(float)
	dow = dt.dt.dayofweek.fillna(0).astype(float)  # 0=Mon
	month = dt.dt.month.fillna(0).astype(float)

	feats = pd.DataFrame(
		{
			'debit_amount': debit,
			'credit_amount': credit,
			'balance_amount': balance,
			'net_amount': net_amount,
			'abs_debit': abs_debit,
			'abs_credit': abs_credit,
			'total_flow': total_flow,
			'is_debit': is_debit,
			'is_credit': is_credit,
			'hour': hour,
			'dow': dow,
			'month': month,
		},
		index=df_base.index,
	)

	expected = _numeric_feature_names_from_scaler()
	if expected is not None:
		# เติมคอลัมน์ที่หายด้วย 0 เพื่อให้ reindex ได้
		for missing in expected:
			if missing not in feats.columns:
				feats[missing] = 0.0
		feats = feats[expected]

	return feats.astype(float)


def _build_features(df: pd.DataFrame):
	"""
	แปลง df อินพุตเป็นเมทริกซ์ฟีเจอร์แบบ csr โดยลำดับฟีเจอร์ = [TXT][CAT][NUM]
	ตรวจมิติเทียบ meta['total_input_dim'] (ถ้ามี) หรือ model.input_shape[-1]
	"""
	from scipy.sparse import csr_matrix, hstack

	_load_artifacts()
	base = _coerce_base(df)

	# 1) TEXT (TF-IDF) — มิติควรเท่ากับ meta['tfidf_dim'] (ถ้ามี)
	X_txt = _cached_tfidf.transform(base['description_text'])
	tfidf_dim_cur = X_txt.shape[1]
	tfidf_dim_meta = (_cached_meta or {}).get('tfidf_dim', tfidf_dim_cur)
	if tfidf_dim_cur != tfidf_dim_meta:
		raise ValueError(
			'TF-IDF dim mismatch: got '
			f'{tfidf_dim_cur}, expected {tfidf_dim_meta}. '
			'Use the exact pre_tfidf.joblib from training.'
		)

	# 2) NUMERIC — reindex ตาม meta/scaler -> scale -> csr
	num_cols = _numeric_feature_names_from_scaler()
	if not num_cols:
		raise ValueError(
			'Numeric feature list not found. Provide it via '
			"model_meta.json['numeric_features'] or scaler.feature_names_in_, "
			'or pre_numeric_features.json.'
		)
	num_df = _build_numeric_features(base)
	for c in num_cols:
		if c not in num_df.columns:
			num_df[c] = 0.0
	num_df = num_df[num_cols]  # ล็อกลำดับ
	X_num_scaled = _cached_scaler.transform(num_df.values)
	X_num_csr = csr_matrix(X_num_scaled)

	# 3) CATEGORICAL — ใช้ vocab แยกเป็น tx_code และ channel
	# แยกจาก code_channel_raw (เช่น "X1/ENET")
	sp = base['code_channel_raw'].str.split('/', n=1, expand=True)
	tx_series = sp[0].fillna('').astype(str).str.upper()
	ch_series = sp[1].fillna('').astype(str).str.upper()

	# ดึง vocab ตามลำดับ: model_meta.json > pre_categ_vocab.json
	tx_vocab = None
	ch_vocab = None
	if _cached_meta:
		cat_meta = _cached_meta.get('categorical') or {}
		tx_vocab = cat_meta.get('tx_code') or cat_meta.get('tx_vocab')
		ch_vocab = cat_meta.get('channel') or cat_meta.get('ch_vocab')

	if (tx_vocab is None or ch_vocab is None) and _cached_vocab:
		# pre_categ_vocab.json ของคุณเป็น {"tx_vocab":[...], "ch_vocab":[...]}
		tx_vocab = tx_vocab or _cached_vocab.get('tx_vocab')
		ch_vocab = ch_vocab or _cached_vocab.get('ch_vocab')

	if not tx_vocab or not ch_vocab:
		raise ValueError(
			"Categorical vocab not found. Provide 'tx_vocab' and 'ch_vocab' "
			"in pre_categ_vocab.json (or 'categorical.tx_code'/'categorical.channel' "
			'in model_meta.json).'
		)

	# one-hot ตามลำดับ vocab ที่กำหนด
	# one-hot ตามลำดับ vocab ที่กำหนด
	tx_onehots = [
		(tx_series == v).astype(np.float32).values.reshape(-1, 1) for v in tx_vocab
	]
	ch_onehots = [
		(ch_series == v).astype(np.float32).values.reshape(-1, 1) for v in ch_vocab
	]

	# ⬇️ ห่อบรรทัดยาวแก้ E501
	X_tx = (
		np.hstack(tx_onehots)
		if tx_onehots
		else np.zeros((len(df), 0), dtype=np.float32)
	)
	X_ch = (
		np.hstack(ch_onehots)
		if ch_onehots
		else np.zeros((len(df), 0), dtype=np.float32)
	)

	# ❌ ลบบรรทัด re-import ด้านล่างนี้ออก (เป็นต้นเหตุ I001 + F811)
	# from scipy.sparse import csr_matrix, hstack

	X_cat_csr = hstack([csr_matrix(X_tx), csr_matrix(X_ch)], format='csr')

	# 4) CONCAT ตามลำดับเดียวกับตอนเทรน: [TXT][CAT(tx)][CAT(ch)][NUM]
	X = hstack([X_txt, X_cat_csr, X_num_csr], format='csr')

	# 5) ตรวจมิติรวม เทียบ meta.total_input_dim (ถ้ามี) หรือ model.input_shape[-1]
	model = get_model()
	expected_input_dim = (_cached_meta or {}).get(
		'total_input_dim', model.input_shape[-1]
	)
	if X.shape[1] != expected_input_dim:
		raise ValueError(
			'Feature dimension mismatch: got '
			f'{X.shape[1]}, expected {expected_input_dim}. '
			'Ensure TF-IDF, categorical vocab, and numeric feature order '
			'match training exactly.'
		)

	return X


# -----------------------------
# Public inference helpers
# -----------------------------
def predict_proba_df(df: pd.DataFrame) -> pd.Series:
	"""
	รับ DataFrame ที่มีคอลัมน์อย่างน้อย:
	tx_datetime, code_channel_raw, debit_amount, credit_amount, balance_amount,
	description_text
	คืนค่า Series ของ probas (0..1) ตามลำดับ index เดิม
	"""
	X = _build_features(df)
	model = get_model()
	y = model.predict(X, verbose=0).ravel()
	return pd.Series(y, index=df.index)


def predict_one(payload: dict) -> dict:
	"""
	อินพุตแบบ dict (เช่นจาก FastAPI) -> พยากรณ์รายการเดียว
	payload ควรมีคีย์:
	- tx_datetime (เช่น '03/10/2025 14:15' หรือรูปแบบที่ pd.to_datetime เข้าใจ)
	- code_channel_raw (string)
	- debit_amount, credit_amount, balance_amount (numeric)
	- description_text (string)
	"""
	df = pd.DataFrame(
		[
			{
				'tx_datetime': payload.get('tx_datetime'),
				'code_channel_raw': payload.get('code_channel_raw'),
				'debit_amount': payload.get('debit_amount', 0.0),
				'credit_amount': payload.get('credit_amount', 0.0),
				'balance_amount': payload.get('balance_amount', 0.0),
				'description_text': payload.get('description_text', ''),
			}
		]
	)

	prob = float(predict_proba_df(df).iloc[0])

	thr = 0.5
	if _cached_meta and isinstance(_cached_meta, dict):
		thr = float(_cached_meta.get('threshold', 0.5))

	return {
		'score': round(prob, 6),
		'label': int(prob >= thr),
		'threshold': thr,
	}


def summarize_document(probas: pd.Series, threshold: float) -> dict:
	"""
	สรุปผลระดับเอกสาร/ทั้งไฟล์:
	- prediction: 'fraud' ถ้ามีสักแถวที่ proba >= threshold
	- confidence: median ของ probas (ไว้ดูความมั่นใจรวม ๆ)
	- threshold, fraud_count, total
	"""
	is_fraud = probas >= threshold
	return {
		'prediction': 'fraud' if is_fraud.any() else 'normal',
		'confidence': float(probas.median()) if len(probas) else 0.0,
		'threshold': float(threshold),
		'fraud_count': int(is_fraud.sum()),
		'total': int(len(probas)),
	}


__all__ = [
	'get_model',
	'predict_proba_df',
	'predict_one',
	'summarize_document',
]
