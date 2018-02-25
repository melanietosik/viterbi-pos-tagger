TRAIN = "WSJ/WSJ_02-21.pos"

DEV_WORDS = "WSJ/WSJ_24.words"
DEV_POS = "WSJ/WSJ_24.pos"
DEV_OUT = "output/wsj_24.pos"

TEST_WORDS = "WSJ/WSJ_23.words"
# TEST_POS = "WSJ/WSJ_23.pos"
TEST_OUT = "output/wsj_23.pos"

MODEL = "data/hmm_model.txt"
VOCAB = "data/hmm_vocab.txt"
UNK_TOKS = "data/unk_toks.txt"

CONFUSION_MATRIX = "docs/confusion_matrix.csv"

TAGS_WSJ = [
    "#",
    "$",
    "''",
    "(",
    ")",
    ",",
    ".",
    ":",
    "CC",
    "CD",
    "DT",
    "EX",
    "FW",
    "IN",
    "JJ",
    "JJR",
    "JJS",
    "LS",
    "MD",
    "NN",
    "NNP",
    "NNPS",
    "NNS",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "RB",
    "RBR",
    "RBS",
    "RP",
    "SYM",
    "TO",
    "UH",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "WDT",
    "WP",
    "WP$",
    "WRB",
    "``",
]
