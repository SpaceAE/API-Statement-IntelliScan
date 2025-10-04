from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
	status: str = Field(
		..., description='The health status of the API', examples=['healthy']
	)


class PredictResponse(BaseModel):
	prediction: Literal['fraud', 'normal'] = Field(
		..., description='The predicted class label'
	)
	confidence: float = Field(
		..., ge=0, le=1, description='The confidence score of the prediction'
	)
