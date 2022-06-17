from fastapi import FastAPI
from models.DecisionTreeClassifier import PredictionQuery, PredictionResponse
from models.DecisionTreeClassifier import Model, ModelResponse
from models.RuleEngine import RuleChain, RuleChainResponse
from models.DecisionTreeClassifier import ModelCheckResponse
import joblib

app = FastAPI(title="Рекомендации по дереву решений", description="Рекомендации по дереву решений", version="1.0")

@app.post('/model/load',
          tags=["Load Model"],
          response_model=ModelResponse,
          summary="Загрузить правило")

async def load_model(model: Model):
    """
    Загрузить правило в систему и сформировать для него расчетну модель:

    Формат запроса:

    - **model_id**: ID модели для правила
    - **n_features_in_**: число входных параметров модели
    - **features_names**: список с наименованиями входных параметров модели
    - **n_classes_**: число выходных классов (рекомендаций)
    - **classes_**: список с ID выходных классов (рекомендаций)
    - **tree_**: описание структуры дерева решений:

        - **node_count**: число узлов модели
        - **children_left"**: список потомков слева (меньше или равно порогу) для каждого узла (если нет = -1)
        - **children_right**: список потомков слева (больше порога) для каждого узла (если нет = -1)
        - **feature**: сравниваемый с порогом параметр для каждого узла (если нет = -2)
        - **threshold**: порог для каждого узла (если нет = -2)
        - **leaf_class_**: ID класса (рекомендации), соответствующего каждому узлу (если нет = -2)

    Формат ответа:

    - **model_id**: ID модели для правила
    - **tree**: текстовая презентация построенного дерева решений
    - **error_message**: описание ошибки

    Документация по используемому дереву решений:

    - https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    - https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    - python -c "from sklearn import tree; help(tree._tree.Tree)"

    """
    try:
        text_representation = model.build_tree()
        joblib.dump(model.clf, 'db/model'+str(model.model_id)+'.joblib')
        error_message = ""
    except Exception as ex:
        error_message = str(ex)
        text_representation = 'undef'
    return {"model_id": model.model_id, "tree": text_representation, "error_message": error_message}

@app.post("/model/predict/{model_id}",
          tags=["Predictions"],
          response_model=PredictionResponse,
          summary="Получить рекомендацию")

async def get_prediction(model_id: int, query: PredictionQuery):
    """
    Получить рекомендацию для строки входящих данных.

    Формат запроса:

    - **model_id**: ID модели для правила
    - **data**: строка входящих данных, длина строки должна совпадать с числом входных параметров модели

    Формат ответа:

    - **model_id**: ID модели для правила
    - **prediction**: ID класса (рекомендация)
    - **error_message**: описание ошибки

    """
    try:
        clf = joblib.load('db/model'+str(model_id)+'.joblib')
        prediction = clf.predict(dict(query)['data'])
        error_message = ""
    except Exception as ex:
        error_message = str(ex)
        prediction = 0
    return {"model_id": model_id, "prediction": prediction, "error_message": error_message}

@app.get("/model/",
          tags=["Check Model"],
          response_model=ModelCheckResponse,
          summary="Проверка загружена ли модель")

async def check_model(model_id: int):
    """
    Проверить, что модель загружена, и получить число входных параметров модели.

    Формат запроса:

    - **model_id**: ID модели для правила

    Формат ответа:

    - **model_id**: ID модели для правила
    - **result**: True если модель загружена, False если не загружена
    - **n_features_in_**: число входных параметров модели
    - **error_message**: описание ошибки
    """
    try:
        clf = joblib.load('db/model'+str(model_id)+'.joblib')
        clf.predict([[0, 0, 0]])
        exists = True
        error_message = ""
        n_features_in_ = clf.n_features_in_
    except Exception as ex:
        error_message = str(ex)
        exists = False
        n_features_in_ = 0
    return {"model_id": model_id, "result": exists, "n_features_in_": n_features_in_, "error_message": error_message}


@app.post('/model_new/load',
          tags=["Load Model new"],
          response_model=RuleChainResponse,
          summary="Загрузить правило")

async def load_model_new(model: RuleChain):
    """
    Загрузить правило в систему и сформировать для него расчетную модель:
    Формат запроса:

    - **model_id: int** ID модели для правила
    - **n_features_in_: int** число входных параметров модели
    - **features_names: List[str]** список с наименованиями входных параметров модели
    - **n_classes_: int** число выходных классов (рекомендаций)
    - **classes_: List[int]** список с ID выходных классов (рекомендаций)
    - **node_count: int** число операторов (узлов) модели
    - **tree_: List[Operator]**: описание операторов в модели:

        - **operator_id: int** ID оператора внутри модели
        - **operator_type: Literal['condition', 'delay', 'recommendation']** тип оператора внутри модели [условие, накопление, рекомендация]
        - **parent_id: Union[int, None]** ID родителя, только если есть родитель, иначе null (пусто)
        - **feature_id: Union[int, None]** ID параметра, только для типа condition, иначе null (пусто)
        - **threshold_left: Union[float, None]** значение порога слева, только для типа condition, если есть порог слева, иначе null или -Infinity
        - **threshold_right: Union[float, None]** значение порога справа, только для типа condition, если есть порог справа и threshold_right>threshold_left, иначе null или Infinity
        - **children_id: Union[List[int], None]** список ID потомков, только если есть потомки, иначе null (пусто)
        - **leaf_class_: Union[int, None]** ID класса (рекомендации), только для типа recommendation, иначе null (пусто)
        - **delay_sec: Union[float, None]** время накопления в секундах, только для типа delay, иначе null (пусто)

    Формат ответа:

    - **model_id**: ID модели для правила
    - **tree**: текстовая презентация построенного дерева решений
    - **error_message**: описание ошибки

    Документация по используемому дереву решений:

    - https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    - https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    - python -c "from sklearn import tree; help(tree._tree.Tree)"

    """
    try:
        with open('db/rule'+str(model.model_id)+'.json', 'w') as outfile:
            outfile.write(model.json(indent=4))
        joblib.dump(model, 'db/model_new_'+str(model.model_id)+'.joblib')
        text_representation = model.json(indent=4)
        error_message = ""
    except Exception as ex:
        error_message = str(ex)
        text_representation = 'undef'
    return {"model_id": model.model_id, "tree": text_representation, "error_message": error_message}

@app.post("/model_new/predict/{model_id}",
          tags=["Predictions new"],
          response_model=PredictionResponse,
          summary="Получить рекомендацию")

async def get_prediction_new(model_id: int, query: PredictionQuery):
    """
    Получить рекомендацию для строки входящих данных.

    Формат запроса:

    - **model_id**: ID модели для правила
    - **data**: строка входящих данных, длина строки должна совпадать с числом входных параметров модели

    Формат ответа:

    - **model_id**: ID модели для правила
    - **prediction**: ID класса (рекомендация)
    - **error_message**: описание ошибки

    """
    try:
        model = joblib.load('db/model_new_'+str(model_id)+'.joblib')
        print(dict(query)['data'])
        prediction = model.test_predict(dict(query)['data'])
        error_message = ""
    except Exception as ex:
        error_message = str(ex)
        prediction = 0
    return {"model_id": model_id, "prediction": prediction, "error_message": error_message}