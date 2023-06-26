# Lacon classy

## Run model

``` bash
python ./src/cli.py NGRAM_SVC 'pas maca test moist cat'
```

Run pretrained svm model with n-grams as feature vector.

Available model with different feature vecor are shown in Model performance secion.

## Model performance

|Name|F1|Recall|Precision|Accuracy|False Positive|False Negative|True Positive|True Negative|
|---|---|---|---|---|---|---|---|---|
|NGRAM_SVC|0.928|0.913|0.944|0.983|0.006|0.168|0.832|0.994|
|HAND_ENGINEERED_SVC|0.778|0.715|0.910|0.959|0.005|0.565|0.435|0.995|
|HAND_ENGINEERED_AND_NGRAMS_SVC|0.930|0.916|0.944|0.983|0.006|0.162|0.838|0.994|
|NGRAM_ADABOOST|0.912|0.894|0.932|0.980|0.008|0.205|0.795|0.992|
|HAND_ENGINEERED_ADABOOST|0.746|0.680|0.924|0.956|0.003|0.638|0.362|0.997|
|HAND_ENGINEERED_AND_NGRAMS_ADABOOST|0.903|0.886|0.923|0.978|0.009|0.220|0.780|0.991|
|NGRAM_KNN|0.837|0.796|0.894|0.965|0.009|0.398|0.602|0.991|
|HAND_ENGINEERED_KNN|0.773|0.711|0.908|0.958|0.005|0.574|0.426|0.995|
|HAND_ENGINEERED_AND_NGRAMS_KNN|0.861|0.822|0.913|0.970|0.008|0.348|0.652|0.992|

