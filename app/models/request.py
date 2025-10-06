# app/models/request.py
from __future__ import annotations

from typing import Optional

from fastapi import File, Form, UploadFile
from pydantic import BaseModel


class PredictForm(BaseModel):
	"""
	ฟอร์มรับไฟล์ statement:
	- file: รองรับ PDF/CSV/XLSX
	- password: ใช้เฉพาะเมื่อไฟล์ PDF ถูกเข้ารหัส
	"""

	password: Optional[str] = Form(
		None, description='Password for encrypted PDF (if required)'
	)
	file: UploadFile = File(..., description='Upload statement file (PDF/CSV/XLSX)')

	@classmethod
	def as_form(  # ใช้กับ Depends(PredictForm.as_form)
		cls,
		file: UploadFile = File(
			..., description='Upload statement file (PDF/CSV/XLSX)'
		),
		password: Optional[str] = Form(
			None, description='Password for encrypted PDF (if required)'
		),
	) -> 'PredictForm':
		return cls(file=file, password=password)


class PredictQueryParam(BaseModel):
	"""
	Query parameters:
	- only_fraud: แสดงเฉพาะแถวที่เสี่ยง
	"""

	only_fraud: bool = False
