from typing import Literal

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
	prediction: Literal['fraud', 'normal'] = Field(
		..., description='The predicted class label'
	)
	confidence: float = Field(
		..., ge=0, le=1, description='The confidence score of the prediction'
	)
