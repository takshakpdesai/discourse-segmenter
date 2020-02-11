from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

from itester.utils import *


class Tester:
    def __init__(self, model):
        self.model = model

    def test_token(self, test_ids, test_masks, test_segments, test_labels, test_token_ids, test_pos, test_parent,
                   test_tokens,
                   batch_size, default_gpu):
        test_data = TensorDataset(test_ids, test_segments, test_masks, test_labels, test_token_ids, test_pos,
                                  test_parent)
        test_sampler = SequentialSampler(test_data)
        test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
        self.update_documents_tokens(test_data_loader, test_tokens, default_gpu)

    def update_documents_tokens(self, test_data_loader, test_tokens, default_gpu):
        device = torch.device('cuda:'+str(default_gpu))
        self.model.eval()
        for batch in test_data_loader:
            batch = tuple(t.to(device) for t in batch)
            bt_features, bt_segments, bt_masks, bt_relations, bt_ids, bt_pos, bt_parent = batch
            with torch.no_grad():
                predicted_relations = self.model(bt_features, bt_segments, bt_masks, input_pos=bt_pos,
                                                 input_parent=bt_parent)
                tags = get_tags_from_tokens(bt_ids, predicted_relations)
                test_tokens = update_token_label(test_tokens, tags)
        return test_tokens

