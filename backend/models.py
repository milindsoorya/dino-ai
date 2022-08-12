# from __future__ import annotations
from typing import Any, Optional, Union, Dict, List
from pydantic import BaseModel


class TrainApiData(BaseModel):
    model_name: str
    hyperparams: Dict[str, Any]
    epochs: int


class PredictApiData(BaseModel):
    input_image: Any
    model_name: str


class DeleteApiData(BaseModel):
    model_name: str
    model_version: Optional[Union[List[int], int]]  # list | int in python 10
 