import argparse

import api
import model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model", choices=model.model_types(), type=str)
    parser.add_argument("words", type=str)

    args = parser.parse_args()

    model = model.model_type(args.model)
    words = args.words.replace("\n", " ").split(" ")
    clazzes = api.classify(words, model)

    print("word,is_eng")
    for word, clazz in zip(words, clazzes):
        print(f"{word},{clazz}")
