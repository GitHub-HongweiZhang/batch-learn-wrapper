# A wrapper for batch-learn.

## how to use
    > bl_run.sh prefix training_data.csv test_data.csv

prefix is used when caching tmp files.

## model
The original implement of model is from github repo https://github.com/alno/batch-learn,
which is called batch-learn. I add early stopping visualization to the model and 
make it possible to do parameter tuning on the dropout rate when using NN.

## data format
label field1_index1,field2_index2,field3_index1
