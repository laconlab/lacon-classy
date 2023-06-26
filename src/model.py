from typing import List
from enum import Enum

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

import feature_extraction as fe


class ModelType(Enum):
    NGRAM_SVC = 1
    HAND_ENGINEERED_SVC = 2
    HAND_ENGINEERED_AND_NGRAMS_SVC = 3
    NGRAM_ADABOOST = 4
    HAND_ENGINEERED_ADABOOST = 5
    HAND_ENGINEERED_AND_NGRAMS_ADABOOST = 6
    NGRAM_KNN = 7
    HAND_ENGINEERED_KNN = 8
    HAND_ENGINEERED_AND_NGRAMS_KNN = 9


def model_types() -> List[str]:
    return [n.name for n in ModelType]


def model_type(mt: str) -> ModelType:
    mt = mt.strip().upper()
    for n in ModelType:
        if n.name == mt:
            return n
    raise RuntimeError


def new_model(model_type: ModelType) -> Pipeline:
    match model_type:
        case ModelType.NGRAM_SVC:
            return Pipeline([
                ("vectorizer", CountVectorizer(analyzer="char", ngram_range=(1, 5))),
                ("classifier", SVC(C=0.142326, kernel="linear"))
            ])
        case ModelType.HAND_ENGINEERED_SVC:
            return Pipeline([
                ("vectorizer", fe.HandEngineeredExtractor()),
                ("classifier", SVC(C=0.462249, kernel="rbf"))
            ])
        case ModelType.HAND_ENGINEERED_AND_NGRAMS_SVC:
            return Pipeline([
                ("vectorizer", fe.NgramAndHandEngineeredFeatuerExtractor((1, 4))),
                ("classifier", SVC(C=0.1424, kernel="linear"))
            ])
        case ModelType.NGRAM_ADABOOST:
            return Pipeline([
                ("vectorizer", CountVectorizer(analyzer="char", ngram_range=(1, 2))),
                ("classifier", AdaBoostClassifier(n_estimators=379))
            ])
        case ModelType.HAND_ENGINEERED_ADABOOST:
            return Pipeline([
                ("vectorizer", fe.HandEngineeredExtractor()),
                ("classifier", AdaBoostClassifier(n_estimators=10))
            ])
        case ModelType.HAND_ENGINEERED_AND_NGRAMS_ADABOOST:
            return Pipeline([
                ("vectorizer", fe.NgramAndHandEngineeredFeatuerExtractor((1, 2))),
                ("classifier", AdaBoostClassifier(n_estimators=311))
            ])
        case ModelType.NGRAM_KNN:
            return Pipeline([
                ("vectorizer", CountVectorizer(analyzer="char", ngram_range=(1, 2))),
                ("classifier", KNeighborsClassifier(n_neighbors=6, n_jobs=-1, weights="distance"))
            ])
        case ModelType.HAND_ENGINEERED_KNN:
            return Pipeline([
                ("vectorizer", fe.HandEngineeredExtractor()),
                ("classifier", KNeighborsClassifier(n_neighbors=103, n_jobs=-1, weights="distance"))
            ])
        case ModelType.HAND_ENGINEERED_AND_NGRAMS_KNN:
            return Pipeline([
                ("vectorizer", fe.NgramAndHandEngineeredFeatuerExtractor((1, 2))),
                ("classifier", KNeighborsClassifier(n_neighbors=6, n_jobs=-1, weights="distance"))
            ])
        case _:
            raise NotImplementedError
