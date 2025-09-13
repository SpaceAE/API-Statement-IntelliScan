import os
from typing import Optional

import keras
from keras.models import load_model

from app.core.config import settings

_cached_model: Optional[keras.Model] = None


def get_model() -> keras.Model:
	if not os.path.exists(settings.MODEL_PATH):
		raise FileNotFoundError(f'Model file not found at {settings.MODEL_PATH}')

	global _cached_model
	if _cached_model is not None:
		return _cached_model

	_cached_model = load_model(settings.MODEL_PATH)
	return _cached_model


def predict_with_model(input_data: any):
	model = get_model()
	predict = model.predict(input_data)
	return predict
