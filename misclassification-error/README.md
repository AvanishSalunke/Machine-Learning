# Misclassification Error

File: [misclassificationerror.py](misclassificationerror.py). The same criterion is also one of the three in [id3_3.py](../id3/id3_3.py).

## Idea

The simplest impurity measure: if a node predicts its majority class, what fraction of its samples does it get wrong? A good split makes the children more accurate than the parent.

## Formulas

Misclassification error of a node $S$ with class proportions $p_c$:

$$E(S) = 1 - \max_c p_c$$

It is 0 for a pure node and 0.5 for a 50/50 node with two classes.

Weighted error after splitting on feature $A$:

$$E(S, A) = \sum_{v \in A} \frac{|S_v|}{|S|} E(S_v)$$

The code picks the feature with the smallest weighted error.

## How it compares with entropy and Gini

Entropy and Gini are smooth curves, so they react to small improvements in class purity. Misclassification error is a straight line in the class proportion, so it is less sensitive: a split that makes a node purer without changing its majority class can show no improvement at all. Because of this it is rarely used for growing trees, but it is common for pruning and for reporting the final error. Comparing the three on the same data is the purpose of [id3_3.py](../id3/id3_3.py).

## What the code does

The tree builder is the same as in [ID3](../id3/README.md): multi-way splits, each feature used once per path, stop when the node is pure or no features remain, majority class as the leaf, and rules printed at the end. The node value printed is the error, the best score starts at 2 and the lowest weighted error wins.

Data is `mushrooms.csv` with `?` replaced by the mode, edible = 0 and poisonous = 1, split 80/20 with `random_state=42`. It prints accuracy, precision, recall, F1, the confusion matrix and the number of leaves.

## Library version

There is no library file for this one, because scikit-learn does not use misclassification error as a split criterion for classification trees. See [id3tree.py](../id3/id3tree.py) for the library tree.
