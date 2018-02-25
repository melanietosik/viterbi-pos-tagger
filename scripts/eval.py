import sys

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

import settings


def score(key_fp, rep_fp):

    key = open(key_fp, "r").readlines()
    rep = open(rep_fp, "r").readlines()

    assert(len(key) == len(rep))

    true = []
    pred = []

    for i in range(len(key)):

        if not (key[i].split() and rep[i].split()):
            continue

        key[i] = key[i].strip()
        rep[i] = rep[i].strip()

        # Get key POS tags
        key_fields = key[i].split("\t")
        if len(key_fields) != 2:
            print("Format error in key at line {0}: {1}".format(i, key[i]))
            sys.exit(1)
        true.append(key_fields[1])

        # Get response POS tags
        rep_fields = rep[i].split("\t")
        if len(rep_fields) != 2:
            print("Format error in key at line {0}: {1}".format(i, rep[i]))
            sys.exit(1)
        pred.append(rep_fields[1])

    assert(len(true) == len(pred))

    print(classification_report(true, pred, settings.TAGS_WSJ))
    print("Accuracy score: {0}".format(accuracy_score(true, pred)))

    y_true = pd.Series(true, name='true')
    y_pred = pd.Series(pred, name='pred')

    df = pd.crosstab(y_true, y_pred)
    df.to_csv(settings.CONFUSION_MATRIX)


if __name__ == "__main__":

    if len(sys.argv) == 3:
        score(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python eval.py <TRUE .pos> <PREDICTED .pos>")
        sys.exit(1)
