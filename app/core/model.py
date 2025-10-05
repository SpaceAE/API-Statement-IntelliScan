# app/core/model.py
import json
import os
import threading

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf

load_model = tf.keras.models.load_model  # ใช้ผ่าน tf แทนการ import submodule ตรง


ARTIFACT_DIR = os.environ.get(
	'MODEL_DIR',
	'/Users/wysuttida/pattern-project/API-Statement-IntelliScan',
)


NUMERIC_FEATURES = [
	'debit_amount',
	'credit_amount',
	'balance_amount',
	'net_amount',
	'abs_amount',
	'log1p_amount',
	'hour',
	'dayofweek',
	'is_weekend',
	'day',
	'month',
	'year',
]
TEXT_FEATURE = 'description_text'


_lock = threading.Lock()
_STATE = {
	'loaded': False,
	'model': None,
	'scaler': None,
	'tfidf': None,
	'tx_vocab': None,
	'ch_vocab': None,
	'tx_index': None,
	'ch_index': None,
	'threshold': 0.5,
}


def _load_artifacts_once():
	if _STATE['loaded']:
		return
	with _lock:
		if _STATE['loaded']:
			return
		model_path = os.path.join(ARTIFACT_DIR, 'model.h5')
		scaler_path = os.path.join(ARTIFACT_DIR, 'pre_scaler.joblib')
		tfidf_path = os.path.join(ARTIFACT_DIR, 'pre_tfidf.joblib')
		vocab_path = os.path.join(ARTIFACT_DIR, 'pre_categ_vocab.json')
		meta_path = os.path.join(ARTIFACT_DIR, 'model_meta.json')

		if not os.path.exists(model_path):
			raise FileNotFoundError(f'Model not found: {model_path}')

		_STATE['model'] = load_model(model_path, compile=False)
		_STATE['scaler'] = joblib.load(scaler_path)
		_STATE['tfidf'] = joblib.load(tfidf_path)

		with open(vocab_path, 'r') as f:
			voc = json.load(f)
		_STATE['tx_vocab'] = voc['tx_vocab']
		_STATE['ch_vocab'] = voc['ch_vocab']
		_STATE['tx_index'] = {t: i for i, t in enumerate(_STATE['tx_vocab'])}
		_STATE['ch_index'] = {t: i for i, t in enumerate(_STATE['ch_vocab'])}

		if os.path.exists(meta_path):
			with open(meta_path, 'r') as f:
				meta = json.load(f)
			_STATE['threshold'] = float(meta.get('threshold', 0.5))

		_STATE['loaded'] = True


def get_model():
	_load_artifacts_once()
	return _STATE['model']


def _one_hot_from_vocab(series_str: pd.Series, index_map: dict, vocab_size: int):
	arr = series_str.astype(str).map(index_map).to_numpy()
	N = len(arr)
	rows = np.arange(N, dtype=np.int64)
	mask = ~pd.isna(arr)
	cols = arr[mask].astype(np.int64)
	data = np.ones(mask.sum(), dtype=np.float32)
	return sp.csr_matrix(
		(data, (rows[mask], cols)), shape=(N, vocab_size), dtype=np.float32
	)


def _preprocess_row_to_df(payload: dict) -> pd.DataFrame:
	tx_datetime = pd.to_datetime(payload.get('tx_datetime', None), errors='coerce')
	if pd.isna(tx_datetime):
		tx_datetime = pd.Timestamp.utcnow()

	ccr = str(payload.get('code_channel_raw', ''))  # "CODE/CHANNEL"
	sp_ = ccr.split('/', 1)
	tx_code = sp_[0].strip() if len(sp_) > 0 else ''
	channel = sp_[1].strip() if len(sp_) > 1 else ''

	debit = float(payload.get('debit_amount', 0) or 0)
	credit = float(payload.get('credit_amount', 0) or 0)
	balance = float(payload.get('balance_amount', 0) or 0)
	desc = str(payload.get('description_text', '') or '')

	df = pd.DataFrame(
		[
			{
				'tx_datetime': tx_datetime,
				'tx_code': tx_code,
				'channel': channel,
				'debit_amount': debit,
				'credit_amount': credit,
				'balance_amount': balance,
				'description_text': desc,
			}
		]
	)

	df['net_amount'] = df['credit_amount'] - df['debit_amount']
	df['abs_amount'] = df['debit_amount'].abs() + df['credit_amount'].abs()
	df['log1p_amount'] = np.log1p(df['abs_amount'])

	dt = df['tx_datetime']
	df['hour'] = dt.dt.hour
	df['dayofweek'] = dt.dt.dayofweek
	df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
	df['day'] = dt.dt.day
	df['month'] = dt.dt.month
	df['year'] = dt.dt.year

	df['description_text'] = df['description_text'].astype(str)
	return df


def _transform_df_to_X(df: pd.DataFrame):
	sc, tfidf = _STATE['scaler'], _STATE['tfidf']
	tx_index, ch_index = _STATE['tx_index'], _STATE['ch_index']
	tx_vocab, ch_vocab = _STATE['tx_vocab'], _STATE['ch_vocab']

	X_num = sc.transform(df[NUMERIC_FEATURES]).astype(np.float32)
	X_num = sp.csr_matrix(X_num)
	X_tx = _one_hot_from_vocab(df['tx_code'], tx_index, len(tx_vocab))
	X_ch = _one_hot_from_vocab(df['channel'], ch_index, len(ch_vocab))
	X_txt = tfidf.transform(df[TEXT_FEATURE].astype(str)).astype(np.float32)

	return sp.hstack([X_num, X_tx, X_ch, X_txt], format='csr', dtype=np.float32)


def predict_one(payload: dict):
	_load_artifacts_once()
	df = _preprocess_row_to_df(payload)
	X = _transform_df_to_X(df).toarray()
	score = float(_STATE['model'].predict(X, verbose=0).ravel()[0])
	thr = float(_STATE['threshold'])
	label = int(score >= thr)
	return {'score': score, 'label': label, 'threshold': thr}


__all__ = ['get_model', 'predict_one']
