# Title
BERT for Discourse Segmentation

# Description

This is the code for our paper titled "Joint Learning of Syntactic Features helps Discourse Segmentation" accepted at LREC 2020 available on http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.135.pdf . Cite our paper as:


```
@InProceedings{desai-dakle-moldovan:2020:LREC,
  author    = {Desai, Takshak  and  Dakle, Parag Pravin  and  Moldovan, Dan},
  title     = {Joint Learning of Syntactic Features Helps Discourse Segmentation},
  booktitle      = {Proceedings of The 12th Language Resources and Evaluation Conference},
  month          = {May},
  year           = {2020},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {1073--1080},
  url       = {https://www.aclweb.org/anthology/2020.lrec-1.135}
}
```

To run the segmenter, simply type:

python modelrunner.py -train train_file -dev dev_file -test test_file -predicted_answers prediction_file

The training, dev and test files can be obtained from https://github.com/disrpt/sharedtask2019. Please use the .conll files for training and dev. The test file must also be a .conll file. Please note that this project assumes the sentence boundaries are already detected (gold or manually).

# Pre-requisites

You will need the following packages to run our tool:

- pytorch
- pytorch_transformers
