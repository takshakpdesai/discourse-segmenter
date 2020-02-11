import torch.nn as nn


class BERTForTokenClassification(nn.Module):
    def __init__(self, language_model, pos_dict, dep_dict):
        super(BERTForTokenClassification, self).__init__()
        self.language_model = language_model
        self.segment_classifier = nn.Linear(768, 2)
        self.pos_classifier = nn.Linear(768, len(pos_dict))
        self.dep_classifier = nn.Linear(768, len(dep_dict))
        self.loss_fct = nn.CrossEntropyLoss()
        self.num_pos = len(pos_dict)
        self.num_dep = len(dep_dict)

    def forward(self, input_ids, input_masks, input_segments, input_labels=None, input_pos=None, input_parent=None,
                pos=10, dparent=10):
        pooled_output = self.language_model(input_ids, attention_mask=input_masks, token_type_ids=input_segments)[2]
        predicted_relations = self.segment_classifier(pooled_output[12])[:, -1]
        if input_labels is not None:
            predicted_tags = self.pos_classifier(pooled_output[pos])[:, -1]
            predicted_dep_rels = self.dep_classifier(pooled_output[dparent])
            return self.loss_fct(predicted_relations.view(-1, 2), input_labels.view(-1)) + \
                   self.loss_fct(predicted_dep_rels.view(-1, self.num_dep), input_parent.view(-1)) + \
                   self.loss_fct(predicted_tags.view(-1, self.num_pos), input_pos.view(-1))

        return predicted_relations
