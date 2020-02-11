import torch


def get_tags_from_tokens(bt_ids, predicted_relations):
    tags = dict()
    for i, rel in enumerate(predicted_relations):
        relation = rel.max(0)[1]
        tags[bt_ids[i].item()] = relation.item()
    return tags


def update_token_label(test_tokens, tags):
    for i in tags.keys():
        test_token = test_tokens[i]
        test_token.update_label(tags[i])
    return test_tokens


def update_sentence_label(sentence, tags):
    if len(sentence.tokens) > 0:
        sentence.tokens[0].update_label(1)
    for i in range(1, len(sentence.tokens)):
        sentence.tokens[i].update_label(tags[i - 1])
    return sentence


def get_tags_from_sentences(bt_ids, predicted_relations, alignment_dict):
    final_tags = dict()
    tags = torch.max(predicted_relations, dim=-1)[1]
    for i, rel in enumerate(tags):
        sentence = bt_ids[i].item()
        alignment_ids = alignment_dict[sentence]
        final_tags[sentence] = tie_breaker(rel, alignment_ids)
    return final_tags


def tie_breaker(predicted_tags, alignment_ids):
    tags = list()
    for i in alignment_ids:
        tags.append(predicted_tags[i[0]])
    return tags


def update_token_label_from_sentence(test_sentences, tags):
    for s_id in tags.keys():
        predicted_labels = tags[s_id]
        sentence = test_sentences[s_id]
        sentence = update_sentence_label(sentence, predicted_labels)
    return test_sentences
