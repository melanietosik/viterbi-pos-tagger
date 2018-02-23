#!/usr/bin/python
#
# Score key .pos file against response .pos file
#
# Both should consist of lines of the form: token \t tag
# Sentences are separated by empty lines

import sys


def score(key_fp, rep_fp):

    key = open(key_fp, "r").readlines()
    rep = open(rep_fp, "r").readlines()

    if len(key) != len(rep):
        print("Length mismatch between key and response file")
        sys.exit(1)

    correct = 0
    incorrect = 0

    for i in range(len(key)):

        key[i] = key[i].strip()
        rep[i] = rep[i].strip()

        if not key[i]:
            if not rep[i]:
                continue
            else:
                print("Sentence break expected at line {0}".format(i))
                sys.exit(1)

        # Get key POS tags
        key_fields = key[i].split("\t")
        if len(key_fields) != 2:
            print("Format error in key at line {0}: {1}".format(i, key[i]))
            sys.exit(1)
        key_pos = key_fields[1]

        # Get response POS tags
        rep_fields = rep[i].split("\t")
        if len(rep_fields) != 2:
            print("Format error in key at line {0}: {1}".format(i, rep[i]))
            sys.exit(1)
        rep_pos = rep_fields[1]

        if key_pos == rep_pos:
            correct += 1
        else:
            incorrect += 1

    total = correct + incorrect
    print("{0} out of {1} tags correct\n".format(correct, total))
    print("Accuracy: {0}".format(100.0 * correct / total))


if __name__ == "__main__":
    score(sys.argv[1], sys.argv[2])
