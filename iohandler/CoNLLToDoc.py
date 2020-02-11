import io
import sys

from iohandler.utils import *
from libnlp.Document import Document
from libnlp.Sentence import Sentence


class CoNLLReader:

    TRAIN = 0
    DEV = 1
    TEST = 2

    def __init__(self, file, mode):
        self.file_path = file
        self.mode = mode

    def read_file(self, doc_id = 0, sent_id = 0, token_id = 0):
        d, s = None, None
        try:
            documents, sentences, tokens = dict(), dict(), dict()
            doc_no, sent_no, tok_no = doc_id, sent_id, token_id
            reader_object = io.open(self.file_path)
            for line in reader_object.readlines():
                if line.startswith("# "):
                    doc_id = line.split("=")[1].strip()
                    d = Document(doc_id)
                    documents[doc_no] = d
                    doc_no += 1
                    s = Sentence(sent_no)
                    sentences[sent_no] = s
                    sent_no += 1
                    d.link_sentence(s)
                elif line[0].isdigit():
                    t = create_token_from_information(line.split("\t"), tok_no)
                    tokens[tok_no] = t
                    tok_no += 1
                    s.link_token(t)
                else:
                    s = Sentence(sent_no)
                    sentences[sent_no] = s
                    sent_no += 1
                    d.link_sentence(s)
            reader_object.close()
        except IOError:
            sys.exit("File not found at location: " + self.file_path)
        return documents, sentences, tokens
