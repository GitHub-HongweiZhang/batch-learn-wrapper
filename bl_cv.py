# Author: Hongwei Zhang
# Email: hw_zhang@outlook.com

import os
import argparse
import time

import pandas as pd
import numpy as np

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score, f1_score


def read_labels(filename):
    labels = []

    with open(filename, "r") as in_f:
        for line in in_f:
            labels.append(float(line.split(" ")[0].strip()))

    return np.array(labels)


def find_threshold(label, pred):
    best_score = f1_score(label, pred > 0.5, average="micro")
    best_threshold = 0.5

    for threshold in np.linspace(0, 1, 1000):
        score = f1_score(label, pred > threshold, average="micro")
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def cv(args, default_param, grid):
    result = []
    for param in ParameterGrid(grid):
        cv_param = default_param.copy()
        cv_param.update(param)

        scores = []

        for fold in range(1, args.kfold + 1):
            cv_train_filename = (args.cache + "/" + "_".join(
                [args.prefix, "cv", "train", "fold", str(fold)]))
            cv_valid_filename = (args.cache + "/" + "_".join(
                [args.prefix, "cv", "valid", "fold", str(fold)]))
            pred_filename = (args.cache + "/" + "_".join(
                [args.prefix, "cv", "valid",
                    "fold", str(fold), "pred"]) + ".csv")

            os.system("./batch-learn ffm --train {} --val {} \
                      --test {} --pred {} --seed {} --threads {} \
                      --eta {} --lambda {} \
                      --epochs {}".format(
                          cv_train_filename,
                          cv_valid_filename,
                          cv_valid_filename,
                          pred_filename,
                          args.seed,
                          args.threads,
                          cv_param["eta"],
                          cv_param["l2_reg"],
                          cv_param["epochs"]))

            label = read_labels(cv_valid_filename + ".csv")
            pred = read_labels(pred_filename)

            try:
                scores.append(roc_auc_score(label, pred, average="micro"))
            except ValueError:
                continue

        cv_param["auc"] = np.mean(scores)
        result.append(cv_param)

    result_df = pd.DataFrame(result).sort_values(
            by="auc", ascending=False).reset_index()
    best_param = result_df.loc[0, [
        "eta", "l2_reg", "epochs"]].to_dict()

    return best_param, result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create cv")
    parser.add_argument("-k", "--kfold", type=int, default=5, help="kfold")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="seed for batch-learn")
    parser.add_argument("-t", "--threads", type=int, default=8,
                        help="num of threads for batch-learn")
    parser.add_argument("-i", "--cache", type=str, required=True,
                        help="cache dir")
    parser.add_argument("-p", "--prefix", type=str, required=True,
                        help="prefix of cv filename")
    args = parser.parse_args()

    grid = {"l2_reg": [1.99999995e-03]}

    default_param = {
            "l2_reg": 1.99999995e-05,
            "eta": 0.00199999996, "epochs": 60}

    start_time = time.time()

    # 5 folds cv
    best_param, result_df = cv(args, default_param, grid)

    train_filename = args.cache + "/" + "_".join(
            [args.prefix, "train"])

    test_filename = args.cache + "/" + "_".join(
            [args.prefix, "test"])

    pred_probe_filename = (args.cache + "/" + "_".join(
        [args.prefix, "test_pred_probe"]) + ".csv")

    # predict test data
    os.system("./batch-learn ffm --train {} \
              --test {} --pred {} --seed {} --threads {} \
              --eta {} --lambda {} \
              --epochs {}".format(
                  train_filename,
                  test_filename,
                  pred_probe_filename,
                  args.seed,
                  args.threads,
                  best_param["eta"],
                  best_param["l2_reg"],
                  best_param["epochs"].astype(np.int)))

    print("---------------summary-----------------")
    print("cv done within {} seconds".format(time.time() - start_time))
    print("best param based on auc: {}".format(best_param))
    print(result_df)
    print("cv info is also available in file {}".format(
        args.prefix + "_cv_result.csv"))
    print("---------------good job-----------------")
    result_df.to_csv(args.prefix + "_cv_result.csv", index=None)
