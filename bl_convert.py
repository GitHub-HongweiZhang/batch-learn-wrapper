# Author: Hongwei Zhang
# Email: hw_zhang@outlook.com

import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create cv")
    parser.add_argument("-k", "--kfold", type=int, default=5, help="kfold")
    parser.add_argument("-p", "--prefix", type=str, required=True,
                        help="prefix of cv filename")
    parser.add_argument("-i", "--cache", type=str, required=True,
                        help="cache dir")
    args = parser.parse_args()

    for fold in range(1, args.kfold + 1):
        cv_train_filename = (
                args.cache + "/" +
                "_".join([args.prefix, "cv", "train", "fold", str(fold)]))
        cv_valid_filename = (
                args.cache + "/" +
                "_".join([args.prefix, "cv", "valid", "fold", str(fold)]))

        os.system("./batch-learn convert -f ffm -b 24 {} -O {}".format(
            cv_train_filename + ".csv", cv_train_filename))
        os.system("./batch-learn convert -f ffm -b 24 {} -O {}".format(
            cv_valid_filename + ".csv", cv_valid_filename))

    train_filename = args.cache + "/" + "_".join(
            [args.prefix, "train"])

    test_filename = args.cache + "/" + "_".join(
            [args.prefix, "test"])

    os.system("./batch-learn convert -f ffm -b 24 {} -O {}".format(
        train_filename + ".csv", train_filename))
    os.system("./batch-learn convert -f ffm -b 24 {} -O {}".format(
        test_filename + ".csv", test_filename))
