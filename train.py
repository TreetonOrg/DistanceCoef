import os
import subprocess
import argparse
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from flags import Flags

from sklearn.model_selection import cross_val_score, cross_val_predict, ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import mean_squared_error, make_scorer, accuracy_score, \
    confusion_matrix, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer


class Mode(Enum):
    REGRESSOR = 0
    CLASSIFIER = 1


class Classifier(Enum):
    LINEAR_SVC = 0
    DECISION_TREE = 1
    RANDOM_FOREST = 2


class Regressor(Enum):
    LINEAR_REGRESSION = 0
    LINEAR_SVR = 1
    DECISION_TREE = 2
    RANDOM_FOREST = 3


class Features(Flags):
    TREETON_BASE = 1
    TREETON_AGG = 2
    MANUAL = 4
    CHAR_GRAMS = 8


def similarity(i, j):
    return 0.0 if i * j == 0.0 else (1 - abs(i - j))**3


def spearman(a, b):
    return stats.spearmanr(a, b)[0]


def count_vowels(text):
    return len([ch for ch in text if ch in "АЕЁИОУЭЮЯаеёиоуэюя"])


def read_features(dir_name):
    file_names = os.listdir(dir_name)
    file_names = list(filter(lambda x: x.endswith(".meta"), file_names))
    features = [[] for _ in range(len(file_names))]
    for filename in file_names:
        with open(os.path.join(dir_name, filename), "r") as f:
            line_features = list(map(float, f.readlines()[1].split("(")[1].strip()[:-1].split(";")))
            features[int(filename.split(".")[0])] = line_features
    return np.array(features)


def collect_featrues(
        data,
        treeton_output_dir,
        normalization,
        treeton_distrib_dir,
        feature_flags,
        rewrite_cache=False):

    if Features.TREETON_BASE in feature_flags or Features.TREETON_AGG in feature_flags:
        if not os.path.exists(treeton_output_dir):
            os.mkdir(treeton_output_dir)

        left_dir = os.path.join(treeton_output_dir, "left")
        if not os.path.exists(left_dir):
            os.mkdir(left_dir)
        for i, line in enumerate(data["left"]):
            file_name = os.path.join(left_dir, str(i) + ".txt")
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(line)

        right_dir = os.path.join(treeton_output_dir, "right")
        if not os.path.exists(right_dir):
            os.mkdir(right_dir)
        for i, line in enumerate(data["right"]):
            file_name = os.path.join(right_dir, str(i) + ".txt")
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(line)

        # Запуск разборов Тритона.
        right_is_calculated = len(os.listdir(right_dir)) == 2 * data["right"].size
        left_is_calculated = len(os.listdir(left_dir)) == 2 * data["left"].size
        if not left_is_calculated or not right_is_calculated or rewrite_cache:
            subprocess.call(['bash', '-c', "sh run_treeton.sh " +
                             " ".join([left_dir, right_dir, treeton_distrib_dir])])

        left_features = read_features(left_dir)
        right_features = read_features(right_dir)
        features_len = left_features.shape[1]

        left_columns = ["left_" + str(i) for i in range(features_len)]
        right_columns = ["right_" + str(i) for i in range(features_len)]
        sim_columns = ["sim_" + str(i) for i in range(features_len)]

        left_features = pd.DataFrame(left_features, columns=left_columns)
        data = pd.concat([data, left_features], axis=1)
        right_features = pd.DataFrame(right_features, columns=right_columns)
        data = pd.concat([data, right_features], axis=1)

        for feature_index in range(features_len - 3):
            left_column = data["left_" + str(feature_index)].tolist()
            right_column = data["right_" + str(feature_index)].tolist()
            sim_vector = [similarity(l, r) for l, r in zip(left_column, right_column)]
            data["sim_" + str(feature_index)] = sim_vector
        for feature_index in range(features_len - 3, features_len):
            left_column = data["left_" + str(feature_index)].tolist()
            right_column = data["right_" + str(feature_index)].tolist()
            sim_vector = [1 - abs(l-r) for l, r in zip(left_column, right_column)]
            data["sim_" + str(feature_index)] = sim_vector

        if Features.TREETON_AGG in feature_flags:
            data["max_sim_0_10"] = pd.DataFrame(data, columns=sim_columns[:10]).max(axis=1)
            data["max_sim_10_20"] = pd.DataFrame(data, columns=sim_columns[10:20]).max(axis=1)
            data["max_sim_20_30"] = pd.DataFrame(data, columns=sim_columns[20:30]).max(axis=1)
            data["max_sim_50_60"] = pd.DataFrame(data, columns=sim_columns[50:60]).max(axis=1)
            data["max_sim_60_70"] = pd.DataFrame(data, columns=sim_columns[60:70]).max(axis=1)
            data["max_sim_70_73"] = pd.DataFrame(data, columns=sim_columns[70:]).max(axis=1)
        if Features.TREETON_BASE not in feature_flags:
            data.drop(left_columns + right_columns + sim_columns, axis=1, inplace=True)

    if Features.MANUAL in feature_flags:
        data["left_vowels"] = data["left"].apply(count_vowels)
        data["right_vowels"] = data["right"].apply(count_vowels)
        data["left_len"] = data["left"].apply(len)
        data["right_len"] = data["right"].apply(len)
        data["vowels_diff"] = (data["left_vowels"] - data["right_vowels"]).apply(abs)
        data["len_diff"] = (data["left_len"] - data["right_len"]).apply(abs)

    if Features.CHAR_GRAMS in feature_flags:
        char_vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 2))
        char_vectorizer.fit(data["left"] + data["right"])
        left_char_features = char_vectorizer.transform(data["left"]).todense()
        right_char_features = char_vectorizer.transform(data["right"]).todense()
        data = pd.concat([data, pd.DataFrame(left_char_features)], axis=1)
        data = pd.concat([data, pd.DataFrame(right_char_features)], axis=1)

    data["score"] = data["aleksej_score"].map(normalization)
    answer = data["score"].tolist()

    data.drop(["score", "aleksej_score", "left", "right"], axis=1, inplace=True)
    return data, answer


def train(data, answer, mode, model_type):
    if mode == Mode.REGRESSOR:
        if model_type == Regressor.DECISION_TREE:
            model = DecisionTreeRegressor()
        elif model_type == Regressor.LINEAR_REGRESSION:
            model = LinearRegression()
        elif model_type == Regressor.LINEAR_SVR:
            model = LinearSVR()
        elif model_type == Regressor.RANDOM_FOREST:
            model = RandomForestRegressor()
        else:
            raise NotImplementedError("Regressor is not bound")

        metrics = (spearman, mean_squared_error)
    elif mode == Mode.CLASSIFIER:
        if model_type == Classifier.RANDOM_FOREST:
            model = RandomForestClassifier()
        elif model_type == Classifier.DECISION_TREE:
            model = DecisionTreeClassifier()
        elif model_type == Classifier.LINEAR_SVC:
            model = LinearSVC()
        else:
            raise NotImplementedError("Regressor is not bound")

        metrics = (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)
    else:
        assert False

    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    for metric in metrics:
        answer_pred = cross_val_predict(model, data, answer, cv=5)
        if metric == confusion_matrix:
            confusions = confusion_matrix(answer, answer_pred)
            print(metric.__name__)
            print(confusions)
            print()
        elif metric in (precision_score, recall_score, f1_score):
            print(metric.__name__)
            print(metric(answer, answer_pred, average=None))
            print()
        else:
            scoring = make_scorer(metric)
            cv_result = np.array(cross_val_score(model, data, answer, cv=cv, scoring=scoring))
            print("%s CV: %0.3f (+/- %0.3f)" % (metric.__name__, cv_result.mean(), cv_result.std() * 2))
            print()
    model.fit(data, answer)
    return model


def main():
    parser = argparse.ArgumentParser(description='Rhythmic similarity regressors')
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--treeton-distrib-dir", type=str, required=True)
    parser.add_argument("--treeton-temp-dir", type=str, default="treeton")
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--model-type", type=int, default=0)
    parser.add_argument("--features", type=int, default=2)
    args = parser.parse_args()

    mode = Mode(args.mode)
    model_type = Classifier(args.model_type) if mode == Mode.CLASSIFIER else Regressor(args.model_type)

    feature_flags = Features(args.features)

    normalization = lambda x: x
    if mode == Mode.REGRESSOR:
        normalization = lambda x: 1.0 - ((float(x) - 1.0) / 4)

    data_file_names = os.listdir(args.data_dir)
    data_file_names = list(filter(lambda x: x.endswith(".tsv"), data_file_names))

    all_data = []
    for i, file_name in enumerate(data_file_names):
        all_data.append(pd.read_csv(os.path.join(args.data_dir, file_name), sep="\t", header=0))

    data, answer = collect_featrues(
        data=pd.concat(all_data, axis=0, ignore_index=True),
        treeton_output_dir=args.treeton_temp_dir,
        treeton_distrib_dir=args.treeton_distrib_dir,
        normalization=normalization,
        feature_flags=feature_flags)

    print("Counts")
    print(np.histogram(answer, bins=5)[0])
    print()

    model = train(data, answer, mode, model_type)
    if mode == Mode.REGRESSOR or model_type == Regressor.LINEAR_REGRESSION:
        print("Coefficients")
        print(list(model.coef_) + [model.intercept_, ])


main()