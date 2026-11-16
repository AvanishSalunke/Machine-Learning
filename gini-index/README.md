# Gini Index

Files in this folder:

- [giniindex.py](giniindex.py): a decision tree that chooses its splits by the Gini index. It prints the score of every feature at every node and the final if-then rules.

There is no separate library file. In scikit-learn the same rule is `DecisionTreeClassifier(criterion="gini")`, and [id3tree.py](../id3/id3tree.py) shows how to use the library tree (with entropy). This rule is also one of the three compared in [id3_3.py](../id3/id3_3.py).

## In simple words

Pick two things at random from a group, one to be the "item" and one to give it a label. How often would the label be wrong? In a group with only one class, never: the Gini index is 0. In a group that is half and half, half the time: the Gini index is 0.5. So Gini measures **how mixed** a group is, like entropy does, but with simpler maths (no logarithms).

The tree picks the split that leaves the **smallest** mix in the groups. This is the rule used by the CART algorithm.

## Key words

- **Impurity**: how mixed the classes in a group are. Gini, entropy and misclassification error are three ways to measure it.
- **Weighted average**: when combining the groups after a split, a bigger group counts more.

## The maths, step by step

**1. Gini impurity of a group** $S$ with class fractions $p_c$:

$$Gini(S) = 1 - \sum_{c} p_c^2$$

Why this works: the chance that two random picks both have class $c$ is $p_c^2$. Adding over all classes gives the chance that the two picks agree, and $1$ minus that is the chance they disagree. For two classes:

- pure group ($p = 1, 0$): $Gini = 1 - 1 = 0$
- 50/50 group: $Gini = 1 - 0.25 - 0.25 = 0.5$
- 75/25 group: $Gini = 1 - 0.5625 - 0.0625 = 0.375$

**2. Gini of a split.** After splitting on feature $A$ into groups $S_v$, combine them with a weighted average:

$$Gini(S, A) = \sum_{v \in A} \frac{|S_v|}{|S|}\; Gini(S_v)$$

**3. The choice.** Pick the feature with the **lowest** $Gini(S, A)$. (Information gain wants the biggest score, Gini wants the smallest. The "Gini gain" $Gini(S) - Gini(S, A)$ is the same idea turned around.)

## A small worked example

Same six mushrooms as in [ID3](../id3/README.md): 3 edible and 3 poisonous, so $Gini(S) = 0.5$.

**Feature A**: groups of 3 (2 e + 1 p) and 3 (1 e + 2 p)

- Each group: $1 - (\tfrac23)^2 - (\tfrac13)^2 = 0.444$
- $Gini(S, A) = \tfrac36 (0.444) + \tfrac36 (0.444) = 0.444$

**Feature B**: groups of 2 (2 e + 0 p) and 4 (1 e + 3 p)

- First group is pure: $0$
- Second group: $1 - (\tfrac14)^2 - (\tfrac34)^2 = 0.375$
- $Gini(S, B) = \tfrac26 (0) + \tfrac46 (0.375) = 0.250$

Feature B has the lower impurity (0.25 against 0.444), so it is chosen, just as with information gain.

## How the code works

The tree builder is the one described in [ID3](../id3/README.md): multi-way splits on category values, each feature used at most once on a path, a node becomes a leaf when it is pure or no features are left, the leaf label is the majority class, and rules are printed at the end. The differences:

1. The value printed for a node is its Gini impurity.
2. For each feature the code computes the weighted Gini of its groups.
3. The best score starts high (2) and the code keeps the feature with the **lowest** weighted Gini.

Data: `../datasets/mushrooms.csv`, `?` replaced by the most common value, edible = 0, poisonous = 1, 80/20 split with `random_state=42`. It prints accuracy, precision, recall, F1, the error rate, the confusion matrix and the number of leaves.

## Strengths and weaknesses

- Cheaper to compute than entropy because there is no logarithm.
- Usually gives a tree very similar to the entropy tree.
- Like information gain, it favours features with many values.

## Things to try

- Compare the tree and the number of leaves with [gainratio.py](../gain-ratio/gainratio.py) and [id3.py](../id3/id3.py).
- Print the Gini of the root node and see how fast it falls.
