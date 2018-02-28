# Melanie Tosik
# NLP, Viterbi part-of-speech (POS) tagger

from __future__ import division

import os
import string
import sys

import settings
import viterbi

from collections import defaultdict

# Punctuation characters
punct = set(string.punctuation)

# Morphology rules used to assign unknown word tokens
noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
adv_suffix = ["ward", "wards", "wise"]

# Additive smoothing parameter
alpha = 0.001


def generate_vocab(min_cnt=2, train_fp=settings.TRAIN):
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

    # Add newline/unknown word tokens
    vocab.extend([line.strip() for line in open(settings.UNK_TOKS, "r")])

    # Sort
    vocab = sorted(vocab)

    with open(settings.VOCAB, "w") as out:
        for word in vocab:
            out.write("{0}\n".format(word))
    out.close()

    return vocab


def assign_unk(tok):
    """
    Assign unknown word tokens
    """
    # Digits
    if any(char.isdigit() for char in tok):
        return "--unk_digit--"

    # Punctuation
    elif any(char in punct for char in tok):
        return "--unk_punct--"

    # Upper-case
    elif any(char.isupper() for char in tok):
        return "--unk_upper--"

    # Nouns
    elif any(tok.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"

    # Verbs
    elif any(tok.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"

    # Adjectives
    elif any(tok.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"

    # Adverbs
    elif any(tok.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"

    return "--unk--"


def train_model(vocab, train_fp=settings.TRAIN):
    """
    Train part-of-speech (POS) tagger model
    """
    vocab = set(vocab)
    emiss = defaultdict(int)
    trans = defaultdict(int)
    context = defaultdict(int)

    with open(train_fp, "r") as train:

        # Start state
        prev = "--s--"

        for line in train:

            # End of sentence
            if not line.split():
                word = "--n--"
                tag = "--s--"

            else:
                word, tag = line.split()
                # Handle unknown words
                if word not in vocab:
                    word = assign_unk(word)

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

    out.close()

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


def decode_seq(model, vocab):
    """
    Decode sequences
    """
    emiss, trans, context = load_model(model)
    tags = sorted(context.keys())

    # Transition matrix
    A = construct_A(trans, context, tags)

    # Emission matrix
    B = construct_B(emiss, context, tags, vocab)

    tag(tags, vocab, A, B)


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

            # Compute smoothed transition probability
            count = 0
            if ((prev in trans) and (tag in trans[prev])):
                count = trans[prev][tag]

            A[i][j] = (count + alpha) / (context[prev] + alpha * K)

    # Assert proper probability distribution
    for i in range(len(A)):
        row_sum = sum([x for x in A[i]])
        assert(abs(row_sum - 1) < 1e-8)

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

            B[i][j] = (count + alpha) / (context[tag] + alpha * N)

    # Assert proper probability distribution
    for i in range(len(B)):
        row_sum = sum([x for x in B[i]])
        assert(abs(row_sum - 1) < 1e-8)

    return B


def preprocess(vocab, data_fp):
    """
    Preprocess data
    """
    orig = []
    prep = []

    # Read data
    with open(data_fp, "r") as data_file:

        for cnt, word in enumerate(data_file):

            # End of sentence
            if not word.split():
                orig.append(word.strip())
                word = "--n--"
                prep.append(word)
                continue

            # Handle unknown words
            elif word.strip() not in vocab:
                orig.append(word.strip())
                word = assign_unk(word)
                prep.append(word)
                continue

            else:
                orig.append(word.strip())
                prep.append(word.strip())

    assert(len(orig) == len(open(data_fp, "r").readlines()))
    assert(len(prep) == len(open(data_fp, "r").readlines()))

    return orig, prep


def tag(tags, vocab, A, B):
    """
    Tag development/test data
    """
    tagged = []

    # Preprocess data
    data_fp = settings.DEV_WORDS
    if split == "test":
        data_fp = settings.TEST_WORDS
    orig, prep = preprocess(vocab, data_fp)

    # Decode
    decoder = viterbi.Viterbi(vocab, tags, prep, A, B)
    pred = decoder.decode()

    for word, tag in zip(orig, pred):
        tagged.append((word, tag))

    # Write output file
    out_fp = settings.DEV_OUT
    if split == "test":
        out_fp = settings.TEST_OUT

    with open(out_fp, "w") as out:
        for word, tag in tagged:
            if not word:
                out.write("\n")
            else:
                out.write("{0}\t{1}\n".format(word, tag))

    out.close()


def main():
    """
    main()
    """
    global split
    split = sys.argv[1]
    if split not in ["dev", "test"]:
        print('Error: split options are "dev" or "test"')
        sys.exit(1)

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

    print("Decoding {0} split...".format(split))
    decode_seq(model, vocab)

    print("Done")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python3 hmm.py <dev|test>")
        sys.exit(1)
    main()
