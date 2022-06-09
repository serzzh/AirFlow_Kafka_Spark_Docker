from fastapi import Depends, FastAPI, APIRouter, HTTPException, File
from models.bintree import PredictionQuery, PredictionResponse, Model, ModelResponse
import json

fake_db = {}

app = FastAPI(title="Рекомендации по дереву решений", description="Рекомендации по дереву решений", version="1.0")

jsonstring = json.loads('''{
    "model_id": 1,
    "n_features_in_": 3,
    "feature_names": ["Cu50", "pulp_level", "bulb_torn"],
    "n_classes_": 11,
    "classes_": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "node_count": 21,
    "tree_": {
        "node_count": 21,
        "children_left":  [1, 3, 5, 7, 13, 16, 19, 10, 14, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        "children_right": [2, 4, 6, 12, 8, 9, 20, 11, 15, 18, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        "feature": [0, 0, 0, 0, 1, 1, 1, 0, 2, 2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        "threshold": [20.5, 19.5, 21.5, 19, 72, 72, 72, 18.5, 0.5, 0.5, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        "leaf_class_": [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
}''')

#model = jsonable_encoder(jsonstring)
#model.build_tree()
clf=None


@app.post('/model/load',
          tags=["Load Model"],
          response_model=ModelResponse,
          description="Загрузить правило")

async def load_model(model: Model):
    #model = Model.from_dict(mode_json)
    response = model.build_tree()
    fake_db[model.model_id] = model.clf
    return {"tree": response}

@app.post('/model/predict/{model_id}',
          tags=["Predictions"],
          response_model=PredictionResponse,
          description="Получить рекомендацию")

async def get_prediction(query: PredictionQuery, fake_db = Depends(fake_db)):
    data = dict(query)['data']
    print("clf", fake_db)
    prediction = fake_db[0].predict(data)
    return {"prediction": prediction}



#app.include_router(app_home)
#app.include_router(app_iris_predict_v1, prefix='/v1')