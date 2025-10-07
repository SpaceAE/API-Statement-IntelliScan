import json
import os

import joblib
from keras.models import load_model

from app.core.config import settings

ARTIFACT_DIR = os.path.dirname(settings.MODEL_PATH)


def p(name):
	return os.path.join(ARTIFACT_DIR, name)


# 1) โหลด meta
with open(p('model_meta.json'), 'r', encoding='utf-8') as f:
	meta = json.load(f)

print('== META KEYS ==')
print('tfidf_dim:', meta.get('tfidf_dim'))
print('numeric_features:', meta.get('numeric_features'))
print('categorical.tx_code size:', len(meta.get('categorical', {}).get('tx_code', [])))
print('categorical.channel size:', len(meta.get('categorical', {}).get('channel', [])))
print('total_input_dim:', meta.get('total_input_dim'))

# 2) โหลด TF-IDF / Scaler / vocab สำรอง
tfidf = joblib.load(p('pre_tfidf.joblib'))
scaler = joblib.load(p('pre_scaler.joblib'))

try:
	with open(p('pre_categ_vocab.json'), 'r', encoding='utf-8') as f:
		backup_vocab = json.load(f)
except FileNotFoundError:
	backup_vocab = {}

try:
	with open(p('pre_numeric_features.json'), 'r', encoding='utf-8') as f:
		backup_nums = json.load(f)
except FileNotFoundError:
	backup_nums = None

print('\n== DERIVED FROM ARTIFACTS ==')
# TF-IDF dim from transformer
try:
	tfidf_dim_from_vectorizer = len(tfidf.get_feature_names_out())
except Exception:
	tfidf_dim_from_vectorizer = tfidf.vocabulary_ and (
		max(tfidf.vocabulary_.values()) + 1
	)
print('tfidf_dim_from_vectorizer:', tfidf_dim_from_vectorizer)

# numeric features list resolution order (เหมือนในโค้ดคุณ)
numeric_from_meta = meta.get('numeric_features')
numeric_from_scaler = getattr(scaler, 'feature_names_in_', None)
numeric_final = numeric_from_meta or (
	list(numeric_from_scaler) if numeric_from_scaler is not None else backup_nums
)
print('numeric_final(len):', len(numeric_final), '::', numeric_final)

# categorical vocab resolution
tx_vocab = meta.get('categorical', {}).get('tx_code') or backup_vocab.get(
	'tx_vocab', []
)
ch_vocab = meta.get('categorical', {}).get('channel') or backup_vocab.get(
	'ch_vocab', []
)
print('tx_vocab(len):', len(tx_vocab))
print('ch_vocab(len):', len(ch_vocab))

# 3) ตรวจมิติรวม เทียบกับตัวโมเดล
model = load_model(settings.MODEL_PATH, compile=False)
txt = meta.get('tfidf_dim', tfidf_dim_from_vectorizer)
cat = len(tx_vocab) + len(ch_vocab)
num = len(numeric_final)
print('\n== DIM CHECK ==')
print('computed_total =', txt, '+', cat, '+', num, '=', txt + cat + num)
print('meta.total_input_dim =', meta.get('total_input_dim'))
print('model.input_shape[-1] =', model.input_shape[-1])
