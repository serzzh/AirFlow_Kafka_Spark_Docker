from fastapi import APIRouter
from models.schemas.iris import Iris, IrisPredictionResponse
import models.ml.classifier as rule

app_iris_predict_v1 = APIRouter()


@app_iris_predict_v1.post('/rule/predict',
                          tags=["Predictions"],
                          response_model=IrisPredictionResponse,
                          description="Получить рекомендацию")



async def get_prediction(iris: Iris):

    data = dict(iris)['data']
    prediction = rule.clf.predict(data)
    return {"prediction": prediction}
