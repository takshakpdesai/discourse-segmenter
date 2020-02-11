class Document:
    def __init__(self, doc_id):
        self.doc_id = doc_id.replace("\n", "")
        self.sentences = list()

    def link_sentence(self, sentence):
        self.sentences.append(sentence)
