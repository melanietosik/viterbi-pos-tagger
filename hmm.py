# Melanie Tosik
# NLP, Viterbi part-of-speech (POS) tagger

from __future__ import division

import os
import string

import settings
import viterbi

from collections import defaultdict

punct = set(string.punctuation)


def generate_vocab(min_cnt=1, train_fp=settings.TRAIN):
    """
    Generate vocabulary
    """
    vocab = defaultdict(int)

    with open(train_fp, "r") as train:

        for line in train:

            # Ignore empty lines
            if not line.split():
                continue
            tok, tag = line.split("\t")
            vocab[tok] += 1

    # Get list of vobulary words
    vocab = [k for k, v in vocab.items() if v >= min_cnt]

    # Add unknown word/newline token
    vocab.append("--unk--")
    vocab.append("--n--")

    # Sort
    vocab = sorted(vocab)

    with open(settings.VOCAB, "w") as out:
        for word in vocab:
            out.write("{0}\n".format(word))
    out.close()

    return vocab


def train_model(vocab, train_fp=settings.TRAIN):
    """
    Train part-of-speech (POS) tagger model
    """
    vocab = set(vocab)
    emiss = defaultdict(int)
    trans = defaultdict(int)
    context = defaultdict(int)

    with open(train_fp, "r") as train:

        prev = "--s--"
        context[prev] += 1

        for line in train:
            # End of sentence
            if not line.split():
                word = "--n--"
                tag = "--s--"
            else:
                word, tag = line.split()

                # Handle unknown words
                if word not in vocab:
                    word = "--unk--"

            trans[" ".join([prev, tag])] += 1
            emiss[" ".join([tag, word])] += 1
            context[tag] += 1
            prev = tag

    model = []

    with open(settings.MODEL, "w") as out:

        # Write transition counts
        for k, v in trans.items():
            line = "T {0} {1}\n".format(k, v)
            model.append(line)
            out.write(line)

        # Write emission counts
        for k, v in emiss.items():
            line = "E {0} {1}\n".format(k, v)
            model.append(line)
            out.write(line)

        # Write context map
        for tag in context:
            line = "C {0} {1}\n".format(tag, context[tag])
            model.append(line)
            out.write(line)

    return model


def load_model(model):
    """
    Load model
    """
    emiss = defaultdict(dict)
    trans = defaultdict(dict)
    context = defaultdict(dict)

    for line in model:

        if line.startswith("C"):
            type_, tag, count = line.split()
            context[tag] = int(count)
            continue

        type_, tag, x, count = line.split()
        if type_ == "T":
            trans[tag][x] = int(count)
        else:
            emiss[tag][x] = int(count)

    return emiss, trans, context


def decode_seq(model, vocab, data_fp=settings.DEV_WORDS):
    """
    Decode sequences
    """
    emiss, trans, context = load_model(model)
    tags = sorted(context.keys())

    # Transition matrix
    A = construct_A(trans, context, tags)

    # Emission matrix
    B = construct_B(emiss, context, tags, vocab)

    tag(tags, vocab, A, B, data_fp)


def construct_A(trans, context, tags):
    """
    Generate transition matrix A of size K x K
    [A_ij stores the probability of transiting from state s_i to state s_j]
    """
    K = len(tags)
    A = [[0] * K for i in range(K)]

    for i in range(K):
        for j in range(K):
            prev = tags[i]
            tag = tags[j]

            # No smoothing for start probabilities
            if prev == "--s--":
                A[i][j] = 0

            # Compute smoothed transition probability#
            count = 0
            if ((prev in trans) and (tag in trans[prev])):
                count = trans[prev][tag]

            A[i][j] = (count + 1) / (context[prev] + K)

    return A


def construct_B(emiss, context, tags, vocab):
    """
    Generate emission matrix B of size K x N
    [B_ij stores the probability of observing o_j from state s_i]
    """
    K = len(tags)
    N = len(vocab)
    B = [[0] * N for i in range(K)]

    for i in range(K):
        for j in range(N):
            tag = tags[i]
            word = vocab[j]

            # Compute smoothed emission probability
            count = 0
            if word in emiss[tag]:
                count = emiss[tag][word]

            B[i][j] = (count + 1) / (context[tag] + N)

    return B


def preprocess(vocab, data_fp):
    """
    Preprocess data
    """
    data = []

    # Read data
    with open(data_fp, "r") as data_file:

        for cnt, word in enumerate(data_file):

            # End of sentence
            if not word.split():
                word = "--n--"
                data.append(word)
                continue

            # Handle unknown words
            elif word.strip() not in vocab:
                word = "--unk--"
                data.append(word)
                continue

            else:
                data.append(word.strip())

    assert(len(data) == len(open(data_fp, "r").readlines()))
    return data


def tag(tags, vocab, A, B, data_fp):
    """
    Tag development/test data
    """
    tagged = []

    # Preprocess data
    data = preprocess(vocab, data_fp)

    # Decode
    decoder = viterbi.Viterbi(vocab, tags, data, A, B)
    pred = decoder.decode()

    for word, tag in zip(data, pred):
        tagged.append((word, tag))

    # Write output file
    with open(data_fp + ".tagged", "w") as out:
        for word, tag in tagged:
            if word == "--n--":
                out.write("\n")
            else:
                out.write("{0}\t{1}\n".format(word, tag))
    out.close()


def main():
    """
    main()
    """
    if not os.path.isfile(settings.VOCAB):
        print("Generating vocabulary...")
        vocab = generate_vocab()
    else:
        vocab = [line.strip() for line in open(settings.VOCAB, "r")]

    if not os.path.isfile(settings.MODEL):
        print("Training model...")
        model = train_model(vocab)
    else:
        model = [line.strip() for line in open(settings.MODEL, "r")]

    print("Decoding...")
    decode_seq(model, vocab, settings.DEV_WORDS)

    print("Done")


if __name__ == "__main__":
    main()
