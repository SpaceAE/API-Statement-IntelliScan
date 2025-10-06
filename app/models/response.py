from typing import Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
	status: str = Field(
		..., description='The health status of the API', examples=['healthy']
	)


class TransactionResult(BaseModel):
	idx: int = Field(..., description='The index of the transaction')
	tx_datetime: str = Field(..., description='The date and time of the transaction')
	code_channel_raw: str = Field(
		..., description='The raw code channel of the transaction'
	)
	debit_amount: float = Field(..., description='The debit amount of the transaction')
	credit_amount: float = Field(
		..., description='The credit amount of the transaction'
	)
	balance_amount: float = Field(
		..., description='The balance amount after the transaction'
	)
	description_text: str = Field(
		..., description='The description text of the transaction'
	)
	fraud_score: float = Field(
		..., ge=0, le=1, description='The fraud score of the transaction'
	)
	is_fraud: bool = Field(
		..., description='The predicted class label of the transaction'
	)


class PredictResponse(BaseModel):
	prediction: Literal['fraud', 'normal'] = Field(
		..., description='The predicted class label'
	)
	confidence: float = Field(
		..., ge=0, le=1, description='The confidence score of the prediction'
	)
	fraud_count: Optional[int] = Field(
		None, ge=0, description='The number of transactions predicted as fraud'
	)
	transactions: Optional[list[TransactionResult]] = Field(
		None, description='List of transaction results with fraud scores'
	)
