# app/models/request.py
from typing import Optional

from fastapi import File, Form, UploadFile
from pydantic import BaseModel, ConfigDict


class PredictForm(BaseModel):
	# Pydantic v2 ต้องเปิดเพื่อยอมให้มีประเภทอย่าง UploadFile
	model_config = ConfigDict(arbitrary_types_allowed=True)

	file: UploadFile
	password: Optional[str] = None

	@classmethod
	def as_form(
		cls,
		file: UploadFile = File(..., description='PDF statement file'),
		password: Optional[str] = Form(None, description='Password for encrypted PDF'),
	):
		return cls(file=file, password=password)
