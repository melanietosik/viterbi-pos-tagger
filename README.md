# Viterbi part-of-speech (POS) tagger

## Overview

## Run the tagger

```
[viterbi-pos-tagger]$ python3.6 scripts/hmm.py dev
[viterbi-pos-tagger]$ python3.6 scripts/hmm.py test
```

## Evaluation

The evaluation script is implemented in [`scripts/eval.py`](https://github.com/melanietosik/viterbi-pos-tagger/blob/master/eval.py). It prints a [text report](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) showing the main classification metrics on the list of POS tags used in the HMM, as well as the overall [accuracy classification score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html). It also generates a [confusion matrix](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.crosstab.html), which is written to [`doc/confusion_matrix.csv`](https://github.com/melanietosik/viterbi-pos-tagger/blob/master/data/confusion_matrix.csv).

### Run the evaluation script

```
[viterbi-pos-tagger]$ virtualenv -p python3.6 env/
[viterbi-pos-tagger]$ source env/bin/activate
[viterbi-pos-tagger]$ pip install -r requirements.txt
[viterbi-pos-tagger]$ python scripts/eval.py <TRUE .pos> <PREDICTED .pos>
```

To evaluate the results on the development and test set, run the following two commands, respectively:

```
[viterbi-pos-tagger]$ python scripts/eval.py WSJ/WSJ_23.pos output/wsj_23.pos
[viterbi-pos-tagger]$ python scripts/eval.py WSJ/WSJ_24.pos output/wsj_24.pos
```

### Results on the development set

As usual, section 24 of the WSJ corpus is used as the development set. The tagged output file for the development set is `output/wsj_24.pos`. The original corpus files are [`WSJ/WSJ_24.words`](https://github.com/melanietosik/viterbi-pos-tagger/blob/master/WSJ/WSJ_24.words) and [`WSJ/WSJ_24.pos`](https://github.com/melanietosik/viterbi-pos-tagger/blob/master/WSJ/WSJ_24.pos).

Initially, Viterbi decoding with a uniform probability for unknown words and add-one smoothing gave a tagging accuracy of 92.88% on the development set. Adding morphological features to improve the handling of unknown words increased accuracy to a score of 93.13%. Finally, tuning the additive smoothing parameter resulted in a **tagging accuracy score of 95.09% on the development set**.

The table below illustrates a few selected model accuracies at various stages of the tuning process. For more details, please see [`doc/accuracy.md`](https://github.com/melanietosik/viterbi-pos-tagger/blob/master/accuracy.md)

| alpha | accuracy score     |
|------:|-------------------:|
|   1.0 | 0.9313000334824826 |
|   0.5 | 0.9398228472285636 |
|   0.2 | 0.9452104830609077 |
|   0.1 | 0.9477064499436886 |
| 0.001 | 0.9508720664779472 |

Below is the classification report for the tagging accuracy on the development set.

```
             precision    recall  f1-score   support

          #       1.00      1.00      1.00         3
          $       1.00      1.00      1.00       216
         ''       1.00      1.00      1.00       247
          (       1.00      1.00      1.00        54
          )       1.00      1.00      1.00        53
          ,       1.00      1.00      1.00      1671
          .       1.00      1.00      1.00      1337
          :       1.00      1.00      1.00       221
         CC       1.00      0.99      1.00       877
         CD       0.98      0.98      0.98      1054
         DT       0.99      0.99      0.99      2856
         EX       0.97      1.00      0.99        37
         FW       0.29      0.50      0.36         8
         IN       0.99      0.95      0.97      3612
         JJ       0.86      0.94      0.90      2036
        JJR       0.86      0.87      0.87        93
        JJS       0.96      0.94      0.95        53
         LS       1.00      0.60      0.75         5
         MD       1.00      0.98      0.99       339
         NN       0.96      0.93      0.95      4541
        NNP       0.92      0.95      0.93      3216
       NNPS       0.76      0.51      0.61       127
        NNS       0.93      0.96      0.94      2050
        PDT       0.88      0.95      0.91        22
        POS       0.99      0.99      0.99       299
        PRP       0.99      0.99      0.99       538
       PRP$       0.99      1.00      0.99       271
         RB       0.87      0.91      0.89      1044
        RBR       0.73      0.76      0.75        54
        RBS       0.95      0.95      0.95        20
         RP       0.54      0.89      0.67        87
        SYM       1.00      0.80      0.89        10
         TO       1.00      1.00      1.00       805
         UH       0.20      0.25      0.22         4
         VB       0.95      0.93      0.94      1010
        VBD       0.93      0.90      0.92      1020
        VBG       0.91      0.82      0.86       528
        VBN       0.85      0.82      0.84       758
        VBP       0.91      0.89      0.90       422
        VBZ       0.94      0.95      0.94       701
        WDT       0.91      0.95      0.93       123
         WP       0.97      0.99      0.98        90
        WP$       1.00      1.00      1.00         7
        WRB       1.00      0.99      0.99        83
         ``       1.00      1.00      1.00       251

avg / total       0.95      0.95      0.95     32853
```

### Results on the test set

Section 23 of the WSJ corpus is usually reserved for testing. The tagged output file for the test set is `output/wsj_23.pos`. The original corpus file is [`WSJ/WSJ_23.words`](https://github.com/melanietosik/viterbi-pos-tagger/blob/master/WSJ/WSJ_23.words). Note that the original `.pos` file for the test set has not yet been released.

To achieve optimal results on the test set, the additive smoothing alpha parameter is currently set to `alpha = 0.001`. The training file is set to [`WSJ/WSJ_02-21+24.pos`](https://github.com/melanietosik/viterbi-pos-tagger/blob/master/WSJ/WSJ_02-21%2B24.pos), which contains both the original training data and the development data combined.
