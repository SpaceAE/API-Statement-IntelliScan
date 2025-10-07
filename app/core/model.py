from __future__ import annotations

import json
import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from scipy.sparse import csr_matrix, hstack

from app.core.config import settings

# -----------------------------
# Global caches
# -----------------------------
_cached_model = None
_cached_tfidf = None
_cached_scaler = None
_cached_vocab = None
_cached_numlist = None
_cached_meta = None

ARTIFACT_DIR = os.path.dirname(settings.MODEL_PATH)


def _p(name: str) -> str:
	return os.path.join(ARTIFACT_DIR, name)


# -----------------------------
# Load model & artifacts
# -----------------------------
def get_model():
	global _cached_model
	if _cached_model is None:
		if not os.path.exists(settings.MODEL_PATH):
			raise FileNotFoundError(f'Model file not found at {settings.MODEL_PATH}')
		_cached_model = load_model(settings.MODEL_PATH, compile=False)
	return _cached_model


def _load_artifacts():
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
	base = pd.DataFrame(index=df.index)

	for c in ['debit_amount', 'credit_amount', 'balance_amount']:
		base[c] = pd.to_numeric(df.get(c), errors='coerce').fillna(0.0)

	dt = pd.to_datetime(df.get('tx_datetime'), format='%d/%m/%Y %H:%M', errors='coerce')
	base['__dt__'] = dt

	base['code_channel_raw'] = (
		df.get('code_channel_raw').astype(str).fillna('').str.upper()
	)
	base['description_text'] = df.get('description_text').astype(str).fillna('')

	return base


def _build_numeric_features(df_base: pd.DataFrame) -> pd.DataFrame:
	debit = df_base['debit_amount'].astype(float)
	credit = df_base['credit_amount'].astype(float)
	balance = df_base['balance_amount'].astype(float)
	dt = df_base['__dt__']

	net_amount = credit - debit
	abs_amount = debit.abs() + credit.abs()
	log1p_amount = np.log1p(abs_amount)
	hour = dt.dt.hour.fillna(0).astype(float)
	dayofweek = dt.dt.dayofweek.fillna(0).astype(float)
	is_weekend = (dayofweek >= 5).astype(float)
	day = dt.dt.day.fillna(0).astype(float)
	month = dt.dt.month.fillna(0).astype(float)
	year = dt.dt.year.fillna(0).astype(float)

	feats = pd.DataFrame(
		{
			'debit_amount': debit,
			'credit_amount': credit,
			'balance_amount': balance,
			'net_amount': net_amount,
			'abs_amount': abs_amount,
			'log1p_amount': log1p_amount,
			'hour': hour,
			'dayofweek': dayofweek,
			'is_weekend': is_weekend,
			'day': day,
			'month': month,
			'year': year,
		},
		index=df_base.index,
	)

	expected = _numeric_feature_names_from_scaler()
	if expected is not None:
		for missing in expected:
			if missing not in feats.columns:
				feats[missing] = 0.0
		feats = feats[expected]

	# print('DEBUG numeric df columns:', feats.columns.tolist())
	return feats.astype(float)


def _build_features(df: pd.DataFrame):
	_load_artifacts()
	base = _coerce_base(df)

	# TEXT
	X_txt = _cached_tfidf.transform(base['description_text'])
	tfidf_dim_cur = X_txt.shape[1]
	tfidf_dim_meta = (_cached_meta or {}).get('tfidf_dim', tfidf_dim_cur)
	if tfidf_dim_cur != tfidf_dim_meta:
		raise ValueError(
			f'TF-IDF dim mismatch: got {tfidf_dim_cur}, expected {tfidf_dim_meta}'
		)

	# NUMERIC
	num_df = _build_numeric_features(base)
	X_num_scaled = _cached_scaler.transform(num_df)  # ✅ DataFrame, not .values
	X_num_csr = csr_matrix(X_num_scaled)

	# CATEGORICAL
	sp = base['code_channel_raw'].str.split('/', n=1, expand=True)
	tx_series = sp[0].fillna('').astype(str).str.upper()
	ch_series = sp[1].fillna('').astype(str).str.upper()

	tx_vocab = (
		_cached_meta.get('categorical', {}).get('tx_code') if _cached_meta else None
	) or (_cached_vocab.get('tx_vocab') if _cached_vocab else None)

	ch_vocab = (
		_cached_meta.get('categorical', {}).get('channel') if _cached_meta else None
	) or (_cached_vocab.get('ch_vocab') if _cached_vocab else None)

	if not tx_vocab or not ch_vocab:
		raise ValueError('Categorical vocab not found.')

	tx_onehots = [
		(tx_series == v).astype(np.float32).values.reshape(-1, 1) for v in tx_vocab
	]
	ch_onehots = [
		(ch_series == v).astype(np.float32).values.reshape(-1, 1) for v in ch_vocab
	]

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
	X_cat_csr = hstack([csr_matrix(X_tx), csr_matrix(X_ch)], format='csr')

	# CONCAT [TXT][CAT][NUM]
	X = hstack([X_txt, X_cat_csr, X_num_csr], format='csr')

	# Validate input dim
	model = get_model()
	expected_input_dim = (_cached_meta or {}).get(
		'total_input_dim', model.input_shape[-1]
	)
	if X.shape[1] != expected_input_dim:
		raise ValueError(
			f'Feature dimension mismatch: got {X.shape[1]}, '
			f'expected {expected_input_dim}'
		)

	# print('DEBUG final X shape:', X.shape)
	return X


# -----------------------------
# Public inference
# -----------------------------
def predict_proba_df(df: pd.DataFrame) -> pd.Series:
	X = _build_features(df)
	model = get_model()
	y = model.predict(X, verbose=0).ravel()
	# print('DEBUG predict_proba_df output:', y)
	return pd.Series(y, index=df.index)


def predict_one(payload: dict) -> dict:
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
	thr = float((_cached_meta or {}).get('threshold', 0.5))
	return {'score': round(prob, 6), 'label': int(prob >= thr), 'threshold': thr}


def summarize_document(probas: pd.Series, threshold: float) -> dict:
	is_fraud = probas >= threshold
	return {
		'prediction': 'fraud' if is_fraud.any() else 'normal',
		'confidence': float(probas.median()) if len(probas) else 0.0,
		'threshold': float(threshold),
		'fraud_count': int(is_fraud.sum()),
		'total': int(len(probas)),
	}


__all__ = ['get_model', 'predict_proba_df', 'predict_one', 'summarize_document']
