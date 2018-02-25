# Viterbi part-of-speech (POS) tagger

## Overview

## Run the tagger

## Evaluation

The evaluation script is implemented in [`eval.py`](https://github.com/melanietosik/viterbi-pos-tagger/blob/master/eval.py). It prints a [text report](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) showing the main classification metrics on the list of POS tags used in the HMM, as well as the overall [accuracy classification score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html). It also generates a [confusion matrix](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.crosstab.html), which is written to [`data/confusion_matrix.csv`](https://github.com/melanietosik/viterbi-pos-tagger/blob/master/data/confusion_matrix.csv).

### Run the script

```
[viterbi-pos-tagger]$ virtualenv -p python3.6 env/
[viterbi-pos-tagger]$ source env/bin/activate
[viterbi-pos-tagger]$ pip install -r requirements.txt
[viterbi-pos-tagger]$ python eval.py <TRUE .pos> <PREDICTED .pos>
```

To evaluate the results on the development and test set specifically:

```
[viterbi-pos-tagger]$ python eval.py WSJ/WSJ_23.pos wsj_23.pos
[viterbi-pos-tagger]$ python eval.py WSJ/WSJ_24.pos wsj_24.pos
```

### Results on the development set

As usual, section 24 of the WSJ corpus is used as the development set. The tagged output file for the development set is `wsj_24.pos`. The original corpus files are [`WSJ/WSJ_24.words`](https://github.com/melanietosik/viterbi-pos-tagger/blob/master/WSJ/WSJ_24.words) and [`WSJ/WSJ_24.pos`](https://github.com/melanietosik/viterbi-pos-tagger/blob/master/WSJ/WSJ_24.pos).

Initially, Viterbi decoding with a uniform probability for unknown words and add-one smoothing gave a tagging accuracy of 92.88% on the development set. Adding morphological features to improve the handling of unknown words increased accuracy to a score of 93.13%.

Finally, tuning the additive smoothing parameter (as shown below) resulted in a tagging accuracy score of **95.09%** on the development set. 

| alpha | accuracy score     |
|-------|--------------------|
|   1.0 | 0.9313000334824826 |
|   0.5 | 0.9398228472285636 |
|   0.2 | 0.9452104830609077 |
|   0.1 | 0.9477064499436886 |
| 0.001 | 0.9508720664779472 |

For more details on the model accuracies at various stages of the tuning process, see `accuracy.md`.

### Results on the test set

Section 23 of the WSJ corpus is usually reserved for testing. The tagged output file for the test set is `wsj_23.pos`. The original corpus file is [`WSJ/WSJ_23.words`](https://github.com/melanietosik/viterbi-pos-tagger/blob/master/WSJ/WSJ_23.words). Note that the `.pos` file for the test set has not yet been released.

To achieve optimal results on the test set, the additive smoothing alpha parameter is currently set to `alpha = 0.001`. The training file is set to [`WSJ_02-21+24.pos`](https://github.com/melanietosik/viterbi-pos-tagger/blob/master/WSJ/WSJ_02-21%2B24.pos), which contains both the original training data and the development data combined.
