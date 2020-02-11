# Title
BERT for Discourse Segmentation

# Description

This is the code for our paper titled "Joint Learning of Syntactic Features helps Discourse Segmentation" accepted at LREC 2020.

To run the segmenter, simply type:

python modelrunner.py -train <train> -dev <dev> -test <test> -predicted_answers <answers>

The training, dev and test files can be obtained from https://github.com/disrpt/sharedtask2019. Please use the .conll files for training and dev. The test file must also be a .conll file. Please note that this project assumes the sentence boundaries are already detected (gold or manually).

# Pre-requisites

You will need the following packages to run our tool:

- pytorch
- pytorch_transformers
