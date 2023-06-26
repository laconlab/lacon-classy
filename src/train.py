#!python src/train.py

import pickle
from typing import List
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

import data
import model
from constants import MODEL_PATH, DOCS_PATH


@dataclass
class ModelSetup:
    train_x: List[str]
    train_y: List[int]
    test_x: List[str]
    test_y: List[str]
    model_type: model.ModelType


@dataclass
class ReportMetrics:
    name: str
    f1: float
    recall: float
    precision: float
    accuracy: float
    false_positives: float
    false_negative: float
    true_positive: float
    true_negative: float

    def row(self):
        return f"|{self.name}|{self.f1:.3f}|{self.recall:.3f}|" + \
                f"{self.precision:.3f}|{self.accuracy:.3f}|" + \
                f"{self.false_positives:.3f}|{self.false_negative:.3f}|" + \
                f"{self.true_positive:.3f}|{self.true_negative:.3f}|\n"


def run(setup: ModelSetup):
    res = train_test(setup)
    update_docs(res)

    model = train(setup)
    save_model(model, setup.model_type.name)


def train_test(setup: ModelSetup):
    """
    train model on train data and test it on test data.
    It will automaticlly update readme
    """
    pipeline = model.new_model(setup.model_type)
    pipeline.fit(setup.train_x, setup.train_y)

    predictions = pipeline.predict(setup.test_x)
    report = classification_report(setup.test_y, predictions, output_dict=True)
    tn, fp, fn, tp = confusion_matrix(setup.test_y, predictions, normalize="true").ravel()
    return ReportMetrics(
            setup.model_type.name,
            float(report["macro avg"]["f1-score"]),
            float(report["macro avg"]["recall"]),
            float(report["macro avg"]["precision"]),
            float(report["accuracy"]),
            fp, fn, tp, tn
    )


def update_docs(results: ReportMetrics):
    with open(DOCS_PATH, "r+") as f:
        lines = f.readlines()

        start, end = 0, 0
        for i, line in enumerate(lines):
            if line.strip() == "## Model performance":
                start = i + 1  # 1 blank line
            elif start > 0 and start != i and not line.startswith("|"):
                end = i
                break
        assert start != 0 and end != 0, f"{start=} {end=}"

        update = False
        for i, line in enumerate(lines[start:end]):
            if line.startswith(f"|{results.name}"):
                lines[start + i], update = results.row(), True

        if not update:
            lines.insert(end, results.row())

        f.seek(0)
        f.writelines(lines)
        f.truncate()


def train(setup: ModelSetup):
    """
    train on all data (train and test)
    """
    pipeline = model.new_model(setup.model_type)
    x = setup.train_x + setup.test_x
    y = setup.train_y + setup.test_y
    pipeline.fit(x, y)
    return pipeline


def save_model(model: Pipeline, name: str):
    path = f"{MODEL_PATH}/{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    x_tr, x_te, y_tr, y_te = data.read_train_test_data()
    for i, model_type in enumerate(model.ModelType):
        print(model_type)
        setup = ModelSetup(x_tr, y_tr, x_te, y_te, model_type)
        run(setup)
