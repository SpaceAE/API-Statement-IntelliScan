from fastapi import File, Form, UploadFile
from pydantic import BaseModel


class PredictForm(BaseModel):
	password: str = Form(..., description='Password for authenticating statement file')
	file: UploadFile = File(..., description='Upload statement pdf')
