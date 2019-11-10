import numpy as np
import sklearn.externals.six

def iter_lines(lines):
    for line in lines:
        toks = line.split()
        qid = toks[0]
        target = float(toks[4])
        pred = float(toks[5])
        yield (qid, target, pred)

def read_dataset(source):

    if isinstance(source, sklearn.externals.six.string_types):
        source = source.splitlines(True)

    qids, targets, preds = [], [], []
    iter_content = iter_lines(source)
    for qid, target, pred in iter_content:
        qids.append(qid)
        targets.append(target)
        preds.append(pred)

    qids = np.array(qids)
    targets = np.array(targets)
    preds = np.array(preds)

    return (qids, targets, preds)