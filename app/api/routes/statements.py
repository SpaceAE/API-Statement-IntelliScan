from fastapi import APIRouter

router = APIRouter(
	prefix='/statements',
	tags=['statements'],
)


@router.post('/predict')
async def predict():
	return {'message': 'This is endpoint to predict.'}
