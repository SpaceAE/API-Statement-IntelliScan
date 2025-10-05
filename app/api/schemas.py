from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
	message: str = Field(..., description='Human-readable message')
	code: Optional[str] = Field(None, description='App-specific error code')
	stage: Optional[str] = Field(
		None, description='Pipeline stage where error happened'
	)
	hint: Optional[str] = Field(None, description='How to fix')
	context: Optional[Dict] = Field(None, description='Debug context (counts, inputs)')
	traceback: Optional[List[str]] = Field(None, description='Stacktrace (dev only)')
