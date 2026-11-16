# Gain Ratio

Files in this folder:

- [gainratio.py](gainratio.py): a decision tree that chooses its splits by gain ratio. It prints the score of every feature at every node and the final if-then rules.

There is no library version, because scikit-learn has no gain ratio option. The closest library tree is [id3tree.py](../id3/id3tree.py), which uses entropy. The same rule is also one of the three rules compared in [id3_3.py](../id3/id3_3.py).

## In simple words

[ID3](../id3/README.md) has a weakness: it loves features with many values. Imagine a feature that is a unique ID number for each sample. Splitting on it gives one tiny group per sample, and each group is pure. The information gain looks perfect, yet the feature is useless for new data.

Gain ratio fixes this. It takes the information gain and **divides it by a penalty** that gets bigger the more pieces the feature cuts the data into. So a feature must be both informative and not too fragmenting.

## Key words

- **Information gain**: how much the entropy drops after the split (see [ID3](../id3/README.md)).
- **Split information**: how evenly and how finely a feature cuts the data into pieces. Many pieces give a big number.
- **Gain ratio**: information gain divided by split information.

## The maths, step by step

**1. Entropy of a group** $S$, with $p_c$ the fraction of class $c$:

$$H(S) = -\sum_{c} p_c \log_2 p_c$$

**2. Information gain** of feature $A$, where $S_v$ is the group with value $v$:

$$IG(S, A) = H(S) - \sum_{v \in A} \frac{|S_v|}{|S|}\, H(S_v)$$

**3. Split information.** This is the entropy of the *sizes* of the groups, ignoring the classes completely:

$$SplitInfo(S, A) = -\sum_{v \in A} \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}$$

A feature with 2 equal groups has $SplitInfo = 1$. A feature with 8 equal groups has $SplitInfo = 3$. An ID-like feature with $m$ groups of one sample has $SplitInfo = \log_2 m$, which is large.

**4. Gain ratio:**

$$GainRatio(S, A) = \frac{IG(S, A)}{SplitInfo(S, A)}$$

The feature with the highest gain ratio is chosen. If $SplitInfo = 0$ (the feature has only one value at this node, so it does not split anything), the code gives it the score 0.

## A small worked example

Same six mushrooms as in [ID3](../id3/README.md): 3 edible and 3 poisonous.

- Feature A: groups of size 3 and 3 (2 e + 1 p, and 1 e + 2 p). $IG = 0.082$.
- Feature B: groups of size 2 and 4 (2 e + 0 p, and 1 e + 3 p). $IG = 0.459$.

Split information:

- Feature A: $-\tfrac12 \log_2 \tfrac12 - \tfrac12 \log_2 \tfrac12 = 1$
- Feature B: $-\tfrac26 \log_2 \tfrac26 - \tfrac46 \log_2 \tfrac46 = 0.918$

Gain ratios:

- Feature A: $0.082 / 1 = 0.082$
- Feature B: $0.459 / 0.918 = 0.500$

Feature B still wins here. The penalty matters when a feature has many values: a 10-value feature with equal-sized groups has $SplitInfo = \log_2 10 = 3.3$, so with $IG = 0.5$ its gain ratio is only about $0.15$.

## How the code works

The tree builder is exactly the one described in [ID3](../id3/README.md): multi-way splits on category values, each feature used at most once on a path, a node becomes a leaf when it is pure or no features are left, the leaf label is the majority class, and rules are printed at the end. The only difference is the scoring line:

1. For every available feature, for every value $v$, take the rows of that value, compute the class fractions, and add up two sums at the same time: the weighted entropy and the split information.
2. Score $= (H(S) - \text{weighted entropy}) / SplitInfo$, or 0 if $SplitInfo = 0$.
3. Choose the highest score, create a child for each value, and continue.

Data: `../datasets/mushrooms.csv`, `?` replaced by the most common value, edible = 0, poisonous = 1, 80/20 split with `random_state=42`. The script prints accuracy, precision, recall, F1, the error rate, the confusion matrix and the number of leaves.

## Strengths and weaknesses

- Protects against features with many values, which is the main flaw of information gain.
- It can over-favour features with very small $SplitInfo$, because dividing by a small number makes the ratio large. The original C4.5 algorithm handles this by only looking at features whose information gain is at least average.

## Things to try

- Run [id3.py](../id3/id3.py) and this script and compare the first feature chosen at the root.
- Compare the number of leaves of the two trees.
