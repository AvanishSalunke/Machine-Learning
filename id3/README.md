# ID3 Decision Tree

Files: [id3.py](id3.py), [id3_3.py](id3_3.py) and the library version [id3tree.py](id3tree.py)

## Idea

A decision tree classifies a sample by asking a series of questions about its features. Each internal node tests one feature, each branch is one value of that feature, and each leaf gives a class. ID3 builds the tree top down and always picks the feature that reduces uncertainty the most.

## Entropy

Entropy measures how mixed the classes are in a set $S$. With class proportions $p_c$:

$$H(S) = -\sum_{c} p_c \log_2 p_c$$

It is 0 when the set is pure (one class only) and 1 for a perfect 50/50 mix of two classes.

## Information gain

If feature $A$ splits $S$ into subsets $S_v$, one for each value $v$ of $A$, the entropy after the split is the weighted average:

$$H(S \mid A) = \sum_{v \in A} \frac{|S_v|}{|S|} H(S_v)$$

Information gain is the drop in entropy:

$$IG(S, A) = H(S) - H(S \mid A)$$

ID3 chooses the feature with the highest information gain.

## Algorithm

1. Start with all training rows at the root and all features available.
2. If all rows at a node have the same class, or no features are left, make the node a leaf labelled with the majority class.
3. Otherwise compute the information gain of every remaining feature and split on the best one. Create one child for every value of that feature.
4. Remove the used feature from the children and repeat.

This is a multi-way split: a feature with 5 values gives 5 branches, and a feature is used at most once on any path.

## What the code does

- Data: `mushrooms.csv`, where `?` is replaced by the column mode. Target: edible = 0, poisonous = 1. 80/20 split with `random_state=42`.
- [id3.py](id3.py) builds the tree with a node list (each node stores its rows, remaining features, parent, chosen feature, children and label) and prints the gain of every feature at each node.
- After building, it prints the tree as if-then rules, one per leaf.
- At prediction time a row follows the branches. If it meets a feature value that was not seen during training at that node, it stops there and uses that node's majority class.
- [id3_3.py](id3_3.py) runs the same tree builder three times with different split criteria in a loop: gain ratio, Gini index and misclassification error. See [gain-ratio.md](../gain-ratio/README.md), [gini-index.md](../gini-index/README.md) and [misclassification-error.md](../misclassification-error/README.md).

## Library version

[id3tree.py](id3tree.py) uses scikit-learn's `DecisionTreeClassifier(criterion="entropy")`. The criterion is the same, but scikit-learn builds binary trees (each node asks a yes or no question like `feature <= value`), so categories are first converted to integer codes. The tree shape and depth can therefore differ from the multi-way ID3 tree. It prints the tree with `export_text` and also shows the depth and number of leaves.
