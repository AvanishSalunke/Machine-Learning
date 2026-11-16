# Misclassification Error

Files in this folder:

- [misclassificationerror.py](misclassificationerror.py): a decision tree that chooses its splits by the misclassification error. It prints the score of every feature at every node and the final if-then rules.

There is no library version, because scikit-learn does not offer this rule for growing classification trees. See [id3tree.py](../id3/id3tree.py) for the library tree. The rule is also one of the three compared in [id3_3.py](../id3/id3_3.py).

## In simple words

The most direct way to measure how mixed a group is: if the group had to guess one class for everyone, it would pick the most common class. What fraction would it get wrong? That fraction is the misclassification error. A pure group gets everything right (error 0). A half and half group gets half wrong (error 0.5).

The tree picks the split whose groups, after guessing their most common class, make the **fewest mistakes** in total.

## Key words

- **Majority class**: the class with the most samples in a group.
- **Error rate**: the fraction of samples that are not the majority class.

## The maths, step by step

**1. Error of a group** $S$ with class fractions $p_c$:

$$E(S) = 1 - \max_c p_c$$

For two classes: 100/0 gives 0, 75/25 gives 0.25, 50/50 gives 0.5.

**2. Error of a split.** After splitting on feature $A$ into groups $S_v$, take the weighted average:

$$E(S, A) = \sum_{v \in A} \frac{|S_v|}{|S|}\; E(S_v)$$

**3. The choice.** Pick the feature with the **lowest** $E(S, A)$.

## A small worked example

Same six mushrooms as in [ID3](../id3/README.md): 3 edible and 3 poisonous. At the root the error is $1 - \tfrac36 = 0.5$.

**Feature A**: groups of 3 (2 e + 1 p) and 3 (1 e + 2 p)

- Each group: $E = 1 - \tfrac23 = 0.333$
- $E(S, A) = \tfrac36 (0.333) + \tfrac36 (0.333) = 0.333$

**Feature B**: groups of 2 (2 e + 0 p) and 4 (1 e + 3 p)

- First group: $E = 0$
- Second group: $E = 1 - \tfrac34 = 0.25$
- $E(S, B) = \tfrac26 (0) + \tfrac46 (0.25) = 0.167$

Feature B has the lower error, so it is chosen.

## Why it is not the usual choice

Entropy and Gini are smooth curves. Misclassification error is a straight line in the class fraction, and it only looks at the biggest class. So a split that makes a group clearly purer, but does not change which class is the biggest, can show **no improvement at all**.

Example: a node has 80 samples of class 1 and 20 of class 0 (error 0.2). A split gives one child with 40 + 0 and another with 40 + 20. The error is still $\tfrac{40}{100}(0) + \tfrac{60}{100}(\tfrac{20}{60}) = 0.2$: no gain, even though one child became perfectly pure. Gini and entropy would see the improvement. Because of this, error is rarely used to grow trees. It is more common for pruning and for judging a finished tree. Comparing the three rules is the purpose of [id3_3.py](../id3/id3_3.py).

## How the code works

The tree builder is the one described in [ID3](../id3/README.md): multi-way splits on category values, each feature used at most once on a path, a node becomes a leaf when it is pure or no features are left, the leaf label is the majority class, and rules are printed at the end. The differences:

1. The value printed for a node is its error rate.
2. For each feature the code computes $\sum \frac{|S_v|}{|S|}(1 - \max p)$.
3. The best score starts high (2) and the feature with the **lowest** weighted error is chosen.

Data: `../datasets/mushrooms.csv`, `?` replaced by the most common value, edible = 0, poisonous = 1, 80/20 split with `random_state=42`. It prints accuracy, precision, recall, F1, the error rate, the confusion matrix and the number of leaves.

## Strengths and weaknesses

- Simple and easy to understand: it is literally the training error.
- Not sensitive to small improvements in purity, so it can stall or choose poorly.

## Things to try

- Compare the trees from this script, [giniindex.py](../gini-index/giniindex.py) and [id3.py](../id3/id3.py). Do they choose the same first feature on the mushroom data?
