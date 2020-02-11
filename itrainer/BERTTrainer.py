import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from tqdm import trange

from evaluation.seg_eval import get_results
from iohandler.DocToCoNLL import CoNLLWriter
from itester.BERTTester import Tester
from models.BERTModels import BERTForTokenClassification


class Trainer:
    def __init__(self, encoder):
        self.language_model = encoder.model
        self.pos_dict = encoder.pos_dict
        self.dep_dict = encoder.dep_dict

    def train_tokens(self, input_ids, input_masks, input_segments, input_labels, input_token_ids, input_poss,
                     input_parents,
                     batch_size, lr,
                     epochs, default_gpu, gpu_list, path_to_saved_conll_file="/tmp/tmp.conll",
                     test_information=None):
        best_model = None
        best_f_score = 0.0
        device = torch.device('cuda:'+str(default_gpu))
        train_data = TensorDataset(input_ids, input_segments, input_masks, input_labels, input_token_ids, input_poss,
                                   input_parents)
        train_sampler = RandomSampler(train_data)
        train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        model = BERTForTokenClassification(self.language_model, self.pos_dict, self.dep_dict)
        model = nn.DataParallel(model, device_ids=gpu_list)
        model.to(device)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = \
            [{"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
              'weight_decay_rate': 0.01},
             {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}]

        optimizer = Adam(optimizer_grouped_parameters, lr=lr)
        tr_t, tr_l, vd_t, vd_l = [], [], [], []

        for _ in trange(epochs, desc="Epoch"):
            model.train()
            tr_loss = 0.0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_data_loader):
                batch = tuple(item.to(device) for item in batch)
                bt_features, bt_segments, bt_masks, bt_relations, bt_ids, bt_pos, bt_parent = batch
                loss = model(bt_features, bt_segments, bt_masks, input_labels=bt_relations, input_pos=bt_pos,
                             input_parent=bt_parent)
                loss.sum().backward()
                tr_loss += loss.sum().item()
                nb_tr_examples += bt_features.size(0)
                nb_tr_steps += 1
                optimizer.step()
                model.zero_grad()

            tr_t.append(tr_loss)
            tr_l.append(tr_loss / nb_tr_steps)
            print("\nTotal training loss: {}".format(tr_loss))
            print("Train loss: {}".format(tr_loss / nb_tr_steps))

            if test_information is not None:
                tester = Tester(model)
                [test_ids, test_masks, test_segments, test_labels, test_token_ids, test_pos, test_parents, test_tokens,
                 test_documents,
                 goldfile] = test_information
                test_tokens = tester.test_token(test_ids, test_masks, test_segments, test_labels, test_token_ids,
                                                test_pos,
                                                test_parents,
                                                test_tokens,
                                                batch_size, default_gpu)
                test_writer = CoNLLWriter(path_to_saved_conll_file)
                test_writer.write_file(test_documents)
                f_score = get_results(goldfile, path_to_saved_conll_file, flag=True)
                if f_score > best_f_score:
                    best_f_score = f_score
                    best_model = model

        if best_model is not None:
            return best_model
        return model
