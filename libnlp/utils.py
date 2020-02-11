def get_int_for_label(label):
    if 'Begin' in label:
        return 1
    return 0


def get_label_for_int(label):
    if label == 0:
        return "_"
    return "BeginSeg=Yes"
