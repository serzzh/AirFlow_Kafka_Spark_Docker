from pydantic import BaseModel, conlist
from typing import List, Any


class Iris(BaseModel):
    data: List[conlist(float, min_items=3, max_items=3)]


class IrisPredictionResponse(BaseModel):
    prediction: int