from fastapi import Depends, FastAPI, APIRouter, HTTPException, File
from models.bintree import PredictionQuery, PredictionResponse, Model, ModelResponse
import joblib

app = FastAPI(title="Рекомендации по дереву решений", description="Рекомендации по дереву решений", version="1.0")

#@app.on_event('startup')
#async def load_model():
    #clf = joblib.load('clf.joblib')



@app.post('/model/load',
          tags=["Load Model"],
          response_model=ModelResponse,
          description="Загрузить правило")

async def load_model(model: Model):
    text_representation = model.build_tree()
    #joblib.dump(clf, 'clf1.joblib')
    return {"tree": text_representation}

@app.post('/model/predict/{model_id}',
          tags=["Predictions"],
          response_model=PredictionResponse,
          description="Получить рекомендацию")

async def get_prediction(query: PredictionQuery):
    data = dict(query)['data']
    clf = joblib.load('clf.joblib')
    prediction = clf.predict(data)
    return {"prediction": prediction}



#app.include_router(app_home)
#app.include_router(app_iris_predict_v1, prefix='/v1')