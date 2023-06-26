import pickle
from typing import List

from sklearn.pipeline import Pipeline

import model
from constants import MODEL_PATH


def classify(words: List[str], model_type: model.ModelType) -> List[int]:
    model = _load_model(model_type)
    return model.predict(words)


def _load_model(model_type: model.ModelType) -> Pipeline:
    with open(f"{MODEL_PATH}/{model_type.name}.pkl", "rb") as f:
        return pickle.load(f)
