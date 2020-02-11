from libnlp.Token import Token


def create_token_from_information(token_information, tok_no):
    token_id, token_text, token_lemma, upos, xpos, dep_head, dep_rel, label = token_information[0], token_information[
        1], token_information[2], token_information[3], token_information[4], token_information[6], token_information[
                                                                                  7], token_information[9]
    t = Token(token_id, token_text, token_lemma, upos, xpos, dep_head, dep_rel, label, uid=tok_no)
    return t
