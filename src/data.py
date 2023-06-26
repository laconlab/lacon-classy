import csv
from typing import List, Tuple

from sklearn.model_selection import train_test_split

from constants import DATA_PATH, RANDOM_STATE, TEST_SIZE


def _read_data(path: str) -> Tuple[List[str], List[int]]:
    x, y = [], []
    with open(path, "r") as f:
        for i, (word, clazz) in enumerate(csv.reader(f, )):
            if i != 0:
                x.append(word)
                y.append(int(clazz))
    return x, y


def read_train_test_data():
    x, y = _read_data(DATA_PATH)
    return train_test_split(x, y,
                            test_size=TEST_SIZE, random_state=RANDOM_STATE,
                            shuffle=True, stratify=y)
