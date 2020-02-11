class Sentence:
    def __init__(self, sent_id):
        self.sent_id = sent_id
        self.tokens = list()

    def link_token(self, token):
        self.tokens.append(token)

    def get_text(self):
        sent_text = ""
        for token in self.tokens:
            sent_text += token.token_text + " "
        return sent_text
