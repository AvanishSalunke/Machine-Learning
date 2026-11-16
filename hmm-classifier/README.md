# HMM Classifier

Files: [hmmclassifier.py](../hmmclassifier.py), [datasethmmclassifier.py](../datasethmmclassifier.py) and [libhmmclassifier.py](../libhmmclassifier.py)

## Idea

Hidden Markov models can be used as classifiers for sequences. Train one HMM per class, using only the training sequences of that class. To classify a new sequence, ask each model how likely it is to have produced the sequence and pick the best class, weighting by the class prior. This is Bayes' rule with the HMM as the class conditional model:

$$\hat{c} = \arg\max_{c} \; P(c) \, P(O \mid \lambda_c)$$

Working with logarithms avoids numerical underflow:

$$\hat{c} = \arg\max_{c} \; \Big[ \ln P(c) + \ln P(O \mid \lambda_c) \Big]$$

## Ingredients

- **Training each class model** uses the [Baum-Welch algorithm](hmm-baum-welch.md) on all sequences of the class. With several sequences, the expected counts $\gamma$ and $\xi$ are summed over all of them before the update.
- **Scoring a sequence** uses the forward algorithm from [hmm-forward-backward.md](hmm-forward-backward.md) to get $P(O \mid \lambda_c)$.
- **The prior** is the class frequency in the training set, $P(c) = N_c / N$.

## What the code does

Every mushroom is treated as a sequence: its attribute values in column order, each value being a symbol (an integer code).

- Data: `mushrooms.csv` with `?` replaced by the mode. Edible = 0, poisonous = 1. 80/20 split with `random_state=42`.
- For each class, an HMM with 3 hidden states is created with random starting $\pi$, $A$, $B$ (seed 42) and trained for 20 Baum-Welch epochs on all sequences of that class at once. The code runs the forward and backward passes for all sequences together with array operations.
- After each update, a tiny constant is added to $B$ and the rows are renormalised, so no emission probability becomes exactly zero.
- The average negative log likelihood per epoch is stored for both classes and plotted.
- Each test sequence goes through the forward pass of both models. The score of a class is $\ln P(O \mid \lambda_c) + \ln P(c)$ and the larger one wins.
- It prints accuracy, precision, recall, F1 and the confusion matrix.

[datasethmmclassifier.py](../datasethmmclassifier.py) is the general version. It reads `dataset.csv` and uses the column named `class` as the target, converts all columns to integer codes, and trains one HMM for each class found. For more than two classes it reports macro averaged precision, recall and F1. `dataset.csv` is not included in this repository.

## Library version

[libhmmclassifier.py](../libhmmclassifier.py) trains one `hmmlearn.hmm.CategoricalHMM` per class (3 states, 20 iterations, trained on all sequences of the class with the `lengths` argument) and scores each test sequence with `model.score`, adding the log prior.
