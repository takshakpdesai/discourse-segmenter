import torch
from pytorch_transformers import *



class Encoder:
    def __init__(self, max_len):
        self.max_len = max_len
        self.model = BertModel.from_pretrained('bert-base-multilingual-uncased', output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True)
        self.pos_dict = {"": 0}
        self.dep_dict = {"": 0}

    def extend_list(self, l):
        m = l
        m.extend([0] * (self.max_len - len(l)))
        return m

    def get_pos_embedding(self, token):
        if token.xpos not in self.pos_dict:
            self.pos_dict[token.xpos] = len(self.pos_dict)
        return self.pos_dict[token.xpos]

    def get_dep_embedding(self, dep_rel):
        if dep_rel not in self.dep_dict:
            self.dep_dict[dep_rel] = len(self.dep_dict)
        return self.dep_dict[dep_rel]

    def get_representations(self, token, s_tokens):
        w_tokens, p_tokens = ["[CLS]"], [0]
        for i, t in enumerate(s_tokens):
            if t != token:
                t_ = self.tokenizer.tokenize(t.token_text)
            else:
                t_ = ["[MASK]"]
            for j in t_:
                w_tokens.append(j)
        for i, t in enumerate(s_tokens):
            l = len(self.tokenizer.tokenize(t.token_text))
            if t.dep_head == token.uid:
                p_tokens.append(self.get_dep_embedding(t.dep_rel))
                if l > 1:
                    for _ in range(l - 1):
                        p_tokens.append(self.get_dep_embedding('X'))
            else:
                for _ in range(l):
                    p_tokens.append(0)
        w_tokens.append("[SEP]")
        p_tokens.append(0)
        w_tokens.append(" ".join(self.tokenizer.tokenize(token.token_text)))
        return w_tokens, p_tokens

    def encode_token(self, sentences):
        input_ids, input_masks, input_segments, input_labels, input_token_ids, input_poss, parent_ids = list(), list(), list(), list(), list(), list(), list()
        for idx in sentences.keys():
            sentence = sentences[idx]
            for i, token in enumerate(sentence.tokens):
                input_text, parent_rels = self.get_representations(token, sentence.tokens)
                tks = self.tokenizer.convert_tokens_to_ids(input_text)
                parent_id = self.extend_list(parent_rels)
                input_id = self.extend_list(tks)
                input_ids.append(input_id)
                input_masks.append(self.extend_list([1] * (len(input_id))))
                input_segments.append(self.extend_list([0] * (len(input_id))))
                input_labels.append(token.label)
                input_token_ids.append(token.uid)
                input_poss.append(self.get_pos_embedding(token))
                parent_ids.append(parent_id)
        input_ids = torch.tensor(input_ids)
        input_masks = torch.tensor(input_masks)
        input_segments = torch.tensor(input_segments)
        input_labels = torch.tensor(input_labels)
        input_token_ids = torch.tensor(input_token_ids)
        input_poss = torch.tensor(input_poss)
        parent_ids = torch.tensor(parent_ids)
        return input_ids, input_masks, input_segments, input_labels, input_token_ids, input_poss, parent_ids

