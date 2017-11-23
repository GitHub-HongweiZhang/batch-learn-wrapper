#!/bin/bash


prefix=$1
train_filename=$2
test_filename=$3

python3 preprocess.py --encoder_type hash --output_format ffm --prefix ${prefix} --train_filename ${train_filename} --test_filename ${test_filename} --output cache

python3 bl_convert.py --prefix ${prefix} --kfold 5 --cache cache

python3 bl_cv.py --prefix ${prefix} --kfold 5 --cache cache
