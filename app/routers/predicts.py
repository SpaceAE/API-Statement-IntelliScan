from fastapi import APIRouter

router = APIRouter(
	prefix='/predicts',
	tags=['predicts'],
	responses={404: {'message': 'Not found'}},
)


@router.post('/')
async def predict():
	return {'message': 'This is endpoint to predict.'}
