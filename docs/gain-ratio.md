# Gain Ratio (C4.5 style splitting)

File: [gainratio.py](../gainratio.py). The same criterion is also one of the three in [id3_3.py](../id3_3.py).

## Idea

Information gain, used by plain [ID3](id3.md), likes features with many different values. A feature such as an ID number splits the data into many tiny pure groups and gets a very high gain, even though it is useless for new data. Gain ratio fixes this by dividing the gain by a penalty that grows with the number of splits.

## Formulas

Entropy of the node:

$$H(S) = -\sum_{c} p_c \log_2 p_c$$

Information gain of feature $A$:

$$IG(S, A) = H(S) - \sum_{v \in A} \frac{|S_v|}{|S|} H(S_v)$$

Split information, which is the entropy of the partition sizes themselves:

$$SplitInfo(S, A) = -\sum_{v \in A} \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}$$

Gain ratio:

$$GainRatio(S, A) = \frac{IG(S, A)}{SplitInfo(S, A)}$$

A feature with many values has a large split information, which shrinks its ratio. The feature with the highest gain ratio is chosen. If the split information is 0 (the feature has only one value in the node) the code sets the score to 0.

## What the code does

The tree builder is the same as in [ID3](id3.md): multi-way splits on categorical features, each feature used once per path, stop when the node is pure or no features remain, leaf label is the majority class, and rules are printed at the end. Only the scoring line changes. For every candidate feature the code computes the weighted entropy and the split information in the same loop, then scores it with the ratio above.

Data is `mushrooms.csv` with `?` replaced by the mode, edible = 0 and poisonous = 1, 80/20 split with `random_state=42`. The script prints accuracy, precision, recall, F1, error rate, the confusion matrix and the number of leaves.

## Library version

There is no separate library file for gain ratio, because scikit-learn does not offer gain ratio as a split criterion. [id3tree.py](../id3tree.py) is the library tree with entropy.
