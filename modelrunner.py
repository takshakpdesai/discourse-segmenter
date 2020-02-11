import argparse

import torch

from encoders.BERTEncoder import Encoder
from evaluation.seg_eval import get_results
from iohandler.CoNLLToDoc import CoNLLReader
from iohandler.DocToCoNLL import CoNLLWriter
from itester.BERTTester import Tester
from itrainer.BERTTrainer import Trainer

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-train', '--training_file', help="Provide path to training file", type=str, required=True)
argument_parser.add_argument('-dev', '--dev_file', help="Provide path to dev file", type=str, required=True)
argument_parser.add_argument('-test', '--test_file', help="Provide path to test file", type=str, required=True)
argument_parser.add_argument('-max_len', '--max_len', help="Provide maximum sequence length for BERT to process",
                             type=int, default=512)
argument_parser.add_argument('-predicted_answers', '--predicted_answers',
                             help="Provide path to save the predicted test answers", type=str, required=True)
argument_parser.add_argument('-batch_size', '--batch_size', help="Provide the batch size to work with", type=int,
                             default=128)
argument_parser.add_argument('-learning_rate', '--learning_rate', help="Provide the learning rate to work with",
                             type=float, default=3e-5)
argument_parser.add_argument('-epochs', '--epochs', help="Provide the maximum number of training epochs", type=int,
                             default=4)
argument_parser.add_argument('-gpus', '--gpus', help="Provide the number of GPUs you want to work with", type=list,
                             default=[0, 1, 2, 3, 4, 5, 6, 7])
argument_parser.add_argument('-default_gpu', '--default_gpu', help="Provide the default GPU", type=int,
                             default=0)
parsed_args = argument_parser.parse_args()

torch.manual_seed(0)
torch.cuda.set_device(parsed_args.default_gpu)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# read file and get document, sentence and token dictionaries

train_reader1 = CoNLLReader(parsed_args.training_file, CoNLLReader.TRAIN)
documents, sentences, tokens = train_reader1.read_file()

dev_reader = CoNLLReader(parsed_args.dev_file, CoNLLReader.DEV)
dev_documents, dev_sentences, dev_tokens = dev_reader.read_file()

# get encoded features
e = Encoder(parsed_args.max_len)
input_ids, input_masks, input_segments, input_labels, input_token_ids, input_poss, input_parents = e.encode_token(
    sentences)
dev_input_ids, dev_input_masks, dev_input_segments, dev_input_labels, dev_input_token_ids, dev_input_poss, dev_input_parents = e.encode_token(
    dev_sentences)

test_reader = CoNLLReader(parsed_args.test_file, CoNLLReader.TEST)
test_documents, test_sentences, test_tokens = test_reader.read_file()
test_ids, test_masks, test_segments, test_labels, test_token_ids, test_poss, test_parents = e.encode_token(
        test_sentences)

# training begins
trainer = Trainer(e)

model = trainer.train_tokens(input_ids, input_masks, input_segments, input_labels, input_token_ids, input_poss,
                             input_parents,
                             parsed_args.batch_size,
                             parsed_args.learning_rate, parsed_args.epochs, parsed_args.default_gpu, parsed_args.gpus,
                             test_information=[dev_input_ids, dev_input_masks, dev_input_segments, dev_input_labels, dev_input_token_ids, dev_input_poss, dev_input_parents, dev_tokens, dev_documents, parsed_args.dev_file])

# final testing begins

tester = Tester(model)

test_tokens = tester.test_token(test_ids, test_masks, test_segments, test_labels, test_token_ids, test_poss,
                                    test_parents,
                                    test_tokens, parsed_args.batch_size)

test_writer = CoNLLWriter(parsed_args.predicted_answers)
test_writer.write_file(test_documents)
get_results(parsed_args.test_file, parsed_args.predicted_answers, flag=True)
