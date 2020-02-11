import io
import os
import sys


def parse_data(infile, string_input=False):
    if not string_input:
        data = io.open(infile, encoding="utf8").read().strip().replace("\r", "")
    else:
        data = infile.strip()

    tokens = []
    labels = []
    spans = []
    counter = 0
    span_start = -1
    span_end = -1
    for line in data.split("\n"):
        if "\t" in line:  # Token
            fields = line.split("\t")
            label = fields[-1]
            # Ensure correct labeling even if other pipe-delimited annotations found in column 10
            if "BeginSeg=Yes" in label:
                label = "BeginSeg=Yes"
            elif "Seg=B-Conn" in label:
                label = "Seg=B-Conn"
                span_start = counter
            elif "Seg=I-Conn" in label:
                label = "Seg=I-Conn"
                span_end = counter
            else:
                label = "_"
                if span_start > -1:  # Add span
                    if span_end == -1:
                        span_end = span_start
                    spans.append((span_start, span_end))
                    span_start = -1
                    span_end = -1

            tokens.append(fields[1])
            labels.append(label)
            counter += 1

    if span_start > -1 and span_end > -1:  # Add last span
        spans.append((span_start, span_end))

    return tokens, labels, spans


def get_scores(gold_file, pred_file, string_input=False):
    report = ""
    gold_tokens, gold_labels, gold_spans = parse_data(gold_file, string_input)
    pred_tokens, pred_labels, pred_spans = parse_data(pred_file, string_input)

    if os.path.isfile(gold_file):
        doc_name = os.path.basename(gold_file)
    else:
        # Use first few tokens to identify file
        doc_name = " ".join(gold_tokens[0:10]) + "..."

    # Check same number of tokens in both files
    if len(gold_tokens) != len(pred_tokens):
        report += "\nFATAL: different number of tokens detected in gold and pred:\n"
        report += "  o In " + doc_name + ": " + str(len(gold_tokens)) + " gold tokens but " + str(
            len(pred_tokens)) + " predicted tokens\n\n"
        sys.stderr.write(report)
        sys.exit(0)

    # Check if this is EDU or Conn-style data
    if "BeginSeg=Yes" in gold_labels:
        mode = "edu"
        seg_type = "EDUs"
    else:
        mode = "conn"
        seg_type = "conn spans"

    true_positive = 0
    false_positive = 0
    false_negative = 0

    if mode == "edu":
        for i, gold_label in enumerate(gold_labels):
            pred_label = pred_labels[i]
            if gold_label == pred_label:
                if gold_label == "_":
                    continue
                else:
                    true_positive += 1
            else:
                if pred_label == "_":
                    false_negative += 1
                else:
                    if gold_label == "_":
                        false_positive += 1
                    else:  # I-Conn/B-Conn mismatch
                        false_positive += 1
    else:
        for span in gold_spans:
            if span in pred_spans:
                true_positive += 1
            else:
                false_negative += 1
        for span in pred_spans:
            if span not in gold_spans:
                false_positive += 1

    try:
        precision = true_positive / (float(true_positive) + false_positive)
    except Exception as e:
        precision = 0

    try:
        recall = true_positive / (float(true_positive) + false_negative)
    except Exception as e:
        recall = 0

    try:
        f_score = 2 * (precision * recall) / (precision + recall)
    except:
        f_score = 0

    score_dict = {"doc_name": doc_name, "tok_count": len(gold_tokens), "seg_type": seg_type,
                  "gold_seg_count": true_positive + false_negative, "pred_seg_count": true_positive + false_positive,
                  "prec": precision, "rec": recall, "f_score": f_score}
    return score_dict


def get_results(goldfile, predfile, flag=False):
    score_dict = get_scores(goldfile, predfile)

    if flag:
        print("File: " + score_dict["doc_name"])
        print("o Total tokens: " + str(score_dict["tok_count"]))
        print("o Gold " + score_dict["seg_type"] + ": " + str(score_dict["gold_seg_count"]))
        print("o Predicted " + score_dict["seg_type"] + ": " + str(score_dict["pred_seg_count"]))
        print("o Precision: " + str(score_dict["prec"]))
        print("o Recall: " + str(score_dict["rec"]))
        print("o F-Score: " + str(score_dict["f_score"]))
    return score_dict["f_score"]
