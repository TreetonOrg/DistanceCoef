import os
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error, make_scorer


def get_clean_texts(ratings_filename, requests_texts_dir, songs_texts_dir,
                    requests_clean_texts_dir, songs_clean_texts_dir):
    dataset = pd.read_csv(ratings_filename, header=0, delimiter=";")
    requests = {}
    for filename in os.listdir(requests_texts_dir):
        with open(os.path.join(requests_texts_dir, filename), "r", encoding="utf-8") as f:
            request_id = int(filename.split(".")[0])
            text = f.read()
            text = '{"item": ' + text.split("#")[1] + '}'
            clean_text = ""
            for line in json.loads(text)['item']:
                line_text = line['plain']
                clean_text += line_text + "\n"
            requests[request_id] = clean_text.strip()

    songs = {}
    for filename in os.listdir(songs_texts_dir):
        with open(os.path.join(songs_texts_dir, filename), "r", encoding="utf-8") as f:
            song_id = int(filename.split(".")[0])
            text = f.read().split("#")[1].strip()
            songs[song_id] = text
    dataset_requests = dataset["requestId"].tolist()
    targets = [requests[request_id] for request_id in dataset_requests]

    dataset_songs = dataset["songId"].tolist()
    shifts = dataset["shift"].tolist()

    sources = []
    for i, song_id in enumerate(dataset_songs):
        fragment = "\n".join(songs[song_id].split("\n")[shifts[i]:])
        lines = []
        for line in fragment.split("\n"):
            if len(line) != 0:
                lines.append(line)
        fragment = "\n".join(lines)
        fragment = "\n".join(fragment.split("\n")[:len(targets[i].split("\n"))])
        sources.append(fragment)

    os.makedirs(songs_clean_texts_dir, exist_ok=True)
    os.makedirs(requests_clean_texts_dir, exist_ok=True)
    for i in range(len(targets)):
        source = sources[i]
        target = targets[i]
        with open(os.path.join(songs_clean_texts_dir, str(i) + ".txt"), "w") as f:
            f.write(source)
        with open(os.path.join(requests_clean_texts_dir, str(i) + ".txt"), "w") as f:
            f.write(target)


def read_features(dir_name):
    features = dict()
    for filename in os.listdir(dir_name):
        sample_features = []
        if ".meta" not in filename:
            continue
        with open(os.path.join(dir_name, filename), "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                line_feature = [float(num) for num in line.split("(")[1].strip()[:-1].split(";")]
                sample_features.append(line_feature)
        features[int(filename.split(".")[0])] = sample_features
    features = [features[i] for i in range(len(features))]
    return features


def normalization(x):
    return 1 - (float(x) - 1.0) / 4


def spearman(a, b):
    return stats.spearmanr(a, b)[0]


def get_coef(ratings_filename, requests_clean_texts_dir, songs_clean_texts_dir, do_cv=True):
    dataset = pd.read_csv(ratings_filename, header=0, delimiter=";")
    requests_features = read_features(requests_clean_texts_dir)
    songs_features = read_features(songs_clean_texts_dir)
    train_data = []
    vector_length = len(requests_features[0][0])
    for request_features, song_features in zip(requests_features, songs_features):
        num_lines = len(request_features)
        all_sim_vector = [0.0 for _ in range(vector_length)]
        for line_request_features, line_song_features in zip(request_features, song_features):
            sim_vector = [i*j for i, j in zip(line_request_features, line_song_features)]
            for k in range(len(line_request_features) - 3, len(line_request_features)):
                sim_vector[k] = 1 - abs(line_request_features[k]-line_request_features[k])
            all_sim_vector = [all_sim_vector[i] + sim_vector[i] for i in range(len(all_sim_vector))]
        all_sim_vector = [all_sim_vector[i]/num_lines for i in range(len(all_sim_vector))]
        train_data.append([max(all_sim_vector[0:20]), max(all_sim_vector[20:50]),
                           max(all_sim_vector[50:70]), max(all_sim_vector[70:])])
    train_answer = dataset["size"].apply(normalization)
    clf = LinearRegression()
    if do_cv:
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        cv_result_spearman = np.array(cross_val_score(clf, train_data, train_answer, cv=cv, scoring=make_scorer(spearman)))
        cv_result_mse = np.array(cross_val_score(clf, train_data, train_answer, cv=cv, scoring=make_scorer(mean_squared_error)))
        print("Spearman CV: %0.3f (+/- %0.3f)" % (cv_result_spearman.mean(), cv_result_spearman.std() * 2))
        print("MSE CV: %0.3f (+/- %0.3f)" % (cv_result_mse.mean(), cv_result_mse.std() * 2))

    clf.fit(train_data, train_answer)
    return list(clf.coef_) + [clf.intercept_, ]

if __name__ == "__main__":
    ratings_filename = "ratings.csv"
    requests_texts_dir = "requests"
    songs_texts_dir = "songs"
    requests_clean_texts_dir = "request_texts"
    songs_clean_texts_dir = "song_texts"
    # Получение нужны отрывков.
    get_clean_texts(ratings_filename, requests_texts_dir, songs_texts_dir,
                    requests_clean_texts_dir, songs_clean_texts_dir)
    # Запуск разборов Тритона.
    subprocess.call(['bash', '-c', "sh run_treeton.sh " +
                     requests_clean_texts_dir + " " +
                     songs_clean_texts_dir + " " +
                     sys.argv[1]])
    # Получение коэффициентов.
    print("Coefficients:", get_coef(ratings_filename, requests_clean_texts_dir, songs_clean_texts_dir, do_cv=bool(int(sys.argv[2]))))