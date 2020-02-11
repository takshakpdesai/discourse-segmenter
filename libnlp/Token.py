from unidecode import unidecode

from libnlp.utils import *


class Token:
    def __init__(self, token_id, token_text, token_lemma, upos, xpos, dep_head, dep_rel, label, uid=None):
        self.token_id = token_id
        self.token_text = self.decode(token_text)
        self.token_lemma = token_lemma
        self.upos = upos
        self.xpos = xpos
        try:
            self.dep_head = int(dep_head) + uid
        except ValueError:
            self.dep_head = uid
        self.dep_rel = dep_rel
        self.label = get_int_for_label(label.replace("\n", ""))
        self.uid = uid

    def update_label(self, new_label):
        self.label = get_label_for_int(new_label)

    @staticmethod
    def decode(token):
        if "`" in token:
            token = token.replace("`", "'")
        return unidecode(str(token))