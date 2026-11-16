# Gini Index

File: [giniindex.py](../giniindex.py). The same criterion is also one of the three in [id3_3.py](../id3_3.py).

## Idea

The Gini index measures impurity. It is the probability that a randomly chosen sample from a node would be given the wrong label if the label were drawn at random from the class proportions in that node. It is the criterion used by the CART algorithm.

## Formulas

Gini impurity of a node $S$ with class proportions $p_c$:

$$Gini(S) = 1 - \sum_{c} p_c^2$$

It is 0 for a pure node and 0.5 for a 50/50 node with two classes.

After splitting on feature $A$ into subsets $S_v$, the impurity of the split is the weighted average:

$$Gini(S, A) = \sum_{v \in A} \frac{|S_v|}{|S|} Gini(S_v)$$

Unlike information gain, here a lower value is better. The code picks the feature with the smallest weighted Gini.

## What the code does

The tree builder is the same as in [ID3](id3.md): multi-way splits on categorical features, every feature used at most once on a path, stopping when the node is pure or no features remain, majority class as the leaf label, and if-then rules printed at the end.

Differences from the entropy version: the node value printed is the Gini impurity, the best score starts at 2 and the code keeps the feature with the lowest weighted Gini.

Data is `mushrooms.csv` with `?` replaced by the mode, edible = 0 and poisonous = 1, split 80/20 with `random_state=42`. It prints the standard metrics and the number of leaves.

## Library version

There is no separate library file for this one. scikit-learn's `DecisionTreeClassifier(criterion="gini")` would be the equivalent, and [id3tree.py](../id3tree.py) shows how the library tree is used with the entropy criterion. Note that scikit-learn builds binary splits while this code makes multi-way splits.
