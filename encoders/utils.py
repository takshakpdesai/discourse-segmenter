def get_labels(sentence):
    class_list = list()
    for token in sentence.tokens:
        class_list.append(token.label)
    return class_list
