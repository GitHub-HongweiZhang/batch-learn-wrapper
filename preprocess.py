# Author: Hongwei Zhang
# Email: hw_zhang@outlook.com

import os
# import bz2
import errno
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import dump_svmlight_file

from encoder import FFMHashEncoder, SVMHashEncoder, OneHotEncoder


def read_train(filename):
    train_X = list()
    train_y = list()

    with open(filename, "r") as in_f:
        for line in in_f:
            label, features = line.split("\t")
            train_X.append(features.split(","))
            train_y.append(int(float(label)))

    return np.array(train_X), np.array(train_y)


def read_test(filename):
    test_X = list()

    with open(filename, "r") as in_f:
        for line in in_f:
            test_X.append(line.split(","))

    return np.array(test_X)


def save_to_disk(filename, X, y):
    with open(filename, "w") as out_f:
        if y is not None:
            for i in range(len(X)):
                out_f.write("{} {}\n".format(y[i], " ".join(X[i])))
        else:
            for i in range(len(X)):
                out_f.write("{}\n".format(" ".join(X[i])))


def encode_and_save(args,
                    train_filename, original_train_X, train_y,
                    test_filename, original_test_X, test_y):
    enc = None
    if args.encoder_type == "hash":
        if args.output_format == "ffm":
            enc = FFMHashEncoder(args.hash_base, args.hash_offset)
        elif args.output_format == "svm":
            enc = SVMHashEncoder(args.hash_base, args.hash_offset)
        else:
            raise NotImplementedError(
                    "{} output format is not supported".format(
                        args.output_format))
    elif args.encoder_type == "onehot":
        enc = OneHotEncoder()
    else:
        raise NotImplementedError(
                "{} encoder type is not supported".format(
                    args.encoder_type))

    enc.fit(original_train_X)

    train_X = enc.transform(original_train_X)
    test_X = enc.transform(original_test_X)

    if args.encoder_type == "hash":
        save_to_disk(train_filename, train_X, train_y)
        save_to_disk(test_filename, test_X, test_y)
    elif args.encoder_type == "onehot":
        dump_svmlight_file(
                train_X, train_y, train_filename, zero_based=False)
        if test_y is not None:
            dump_svmlight_file(
                    test_X, test_y, test_filename, zero_based=False)
        else:
            # dump_svmlight_file need y, so bad
            dump_svmlight_file(
                    test_X, np.zeros(test_X.shape[0]),
                    f=test_filename, zero_based=False)
    else:
        raise NotImplementedError(
                "{} encoder type is not supported".format(
                    args.encoder_type))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create cv")
    parser.add_argument("-k", "--kfold", type=int, default=5, help="kfold")
    parser.add_argument("-r", "--random_state", type=int, default=0,
                        help="random state for creating cv")
    parser.add_argument("-b", "--hash_base", type=int, default=100000,
                        help="hash base")
    parser.add_argument("-m", "--hash_offset", type=int, default=100,
                        help="hash offset")
    parser.add_argument("-e", "--encoder_type", type=str, default="hash",
                        choices=["hash", "onehot"],
                        help="encoder type, onehot only supports svm format")
    parser.add_argument("-f", "--output_format", type=str, default="ffm",
                        choices=["ffm", "svm"],
                        help="output format when encoder_type is hash")
    parser.add_argument("-p", "--prefix", type=str, required=True,
                        help="prefix of cv filename")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="output dir")
    parser.add_argument("-t", "--train_filename", type=str, required=True,
                        help="filename of training data")
    parser.add_argument("-v", "--test_filename", type=str,
                        help="filename of test data")
    args = parser.parse_args()

    try:
        os.makedirs(args.output)
    except OSError as e:
        # if dir exists, ignore exception
        if e.errno != errno.EEXIST and os.path.isdir(args.output):
            print("cannot create output dir, because {}".format(str(e)))

    original_train_X, train_y = read_train(args.train_filename)

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True,
                          random_state=args.random_state)

    curr_fold = 1
    for train_index, valid_index in skf.split(original_train_X, train_y):
        cv_original_train_X = original_train_X[train_index]
        cv_original_valid_X = original_train_X[valid_index]
        cv_train_y = train_y[train_index]
        cv_valid_y = train_y[valid_index]

        cv_train_filename = args.output + "/" + "_".join(
                [args.prefix, "cv", "train", "fold", str(curr_fold)]) + ".csv"

        cv_valid_filename = args.output + "/" + "_".join(
                [args.prefix, "cv", "valid", "fold", str(curr_fold)]) + ".csv"

        print("encode and save {} and {}".format(
            cv_train_filename, cv_valid_filename))
        encode_and_save(args,
                        cv_train_filename, cv_original_train_X, cv_train_y,
                        cv_valid_filename, cv_original_valid_X, cv_valid_y)

        curr_fold += 1

    if args.test_filename is not None:
        original_test_X = read_test(args.test_filename)
        train_filename = args.output + "/" + "_".join(
                [args.prefix, "train"]) + ".csv"

        test_filename = args.output + "/" + "_".join(
                [args.prefix, "test"]) + ".csv"

        encode_and_save(args,
                        train_filename, original_train_X, train_y,
                        test_filename, original_test_X, None)
