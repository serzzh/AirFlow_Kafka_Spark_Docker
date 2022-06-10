from fastapi import FastAPI
from models.DecisionTreeClassifier import PredictionQuery, PredictionResponse, Model, ModelResponse
import joblib

app = FastAPI(title="Рекомендации по дереву решений", description="Рекомендации по дереву решений", version="1.0")

@app.post('/model/load',
          tags=["Load Model"],
          response_model=ModelResponse,
          description="Загрузить правило")

async def load_model(model: Model):
    text_representation = model.build_tree()
    joblib.dump(model.clf, 'db/model'+str(model.model_id)+'.joblib')
    return {"tree": text_representation}

@app.post('/model/predict/{model_id}',
          tags=["Predictions"],
          response_model=PredictionResponse,
          description="Получить рекомендацию")

async def get_prediction(query: PredictionQuery):
    clf = joblib.load('db/model'+str(dict(query)['model_id'])+'.joblib')
    prediction = clf.predict(dict(query)['data'])
    return {"prediction": prediction}