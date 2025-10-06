from __future__ import annotations

from typing import List

from pydantic import BaseModel


class TxResult(BaseModel):
	idx: int
	tx_datetime: str
	code_channel_raw: str
	debit_amount: float
	credit_amount: float
	balance_amount: float
	description_text: str
	fraud_score: float
	is_fraud: bool


class PredictResponseWithDetails(BaseModel):
	prediction: str
	confidence: float
	threshold: float
	fraud_count: int
	total: int
	transactions: List[TxResult]
	fraud_transactions: List[TxResult] = []
	fraud_indices: List[int] = []
