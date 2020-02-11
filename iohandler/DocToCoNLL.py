import io
import sys


class CoNLLWriter:
    def __init__(self, file):
        self.file_path = file

    def write_file(self, documents):
        try:
            writer_object = io.open(self.file_path, 'w+')
            for idx in documents.keys():
                document = documents[idx]
                writer_object.write("# newdoc id = " + document.doc_id + "\n")
                for sentence in document.sentences:
                    for token in sentence.tokens:
                        writer_object.write(
                            token.token_id + "\t" + token.token_text + "\t_\t_\t_\t_\t_\t_\t_\t" + str(token.label))
                        writer_object.write("\n")
                    writer_object.write("\n")
            writer_object.close()
        except IOError:
            sys.exit("Cannot open file at location: " + self.file_path)
