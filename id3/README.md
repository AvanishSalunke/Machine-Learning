# ID3 Decision Tree

Files in this folder:

- [id3.py](id3.py): ID3 built step by step with numpy. It prints the gain of every feature at every node and the final if-then rules.
- [id3_3.py](id3_3.py): the same tree builder run three times with three other split rules (gain ratio, Gini index, misclassification error), to compare them.
- [id3tree.py](id3tree.py): library version with scikit-learn's `DecisionTreeClassifier`.

## In simple words

Think of the game "20 questions". To guess an animal quickly you ask the question that splits the possibilities best, like "Does it live in water?" and not "Is it a dog?". A decision tree plays the same game with data. At every step it asks about the feature that **removes the most uncertainty** about the class. After a few questions, the group of samples left is (almost) all one class, and the tree can give its answer.

ID3 measures uncertainty with **entropy**, and the drop in uncertainty is called **information gain**.

## Key words

- **Node**: one question in the tree. The first node is the **root**.
- **Branch**: one answer to a question (one value of the feature).
- **Leaf**: a node that gives the final class and asks nothing more.
- **Entropy**: how mixed up a group is. 0 means all the same class.
- **Information gain**: how much the entropy drops after splitting on a feature.
- **Pure node**: a node where every sample has the same class.

## The maths, step by step

**1. Entropy.** For a group of samples $S$, let $p_c$ be the fraction that belongs to class $c$:

$$H(S) = -\sum_{c} p_c \log_2 p_c$$

Read it like this: a class that is rare is "surprising", so $-\log_2 p_c$ is big for small $p_c$. Entropy is the average surprise. For two classes:

- all one class ($p = 1$ and $0$): $H = 0$, no uncertainty
- a 50/50 mix: $H = 1$, the maximum uncertainty
- a 90/10 mix: $H = 0.47$

(We treat $0 \log_2 0$ as 0.)

**2. Entropy after a split.** If we split $S$ on feature $A$, we get one smaller group $S_v$ for each value $v$ of $A$. We take the entropy of each small group and average them, giving more weight to bigger groups:

$$H(S \mid A) = \sum_{v \in A} \frac{|S_v|}{|S|}\, H(S_v)$$

**3. Information gain.** The drop in entropy:

$$IG(S, A) = H(S) - H(S \mid A)$$

ID3 splits on the feature with the highest information gain.

**4. The algorithm.**

1. Put all training rows in the root node. All features are available.
2. If the node is pure, or no features are left, make it a leaf. Its class is the most common class among its rows.
3. Otherwise compute $IG$ for each available feature and pick the best one.
4. Make one child node for every value of that feature, and give each child the rows with that value.
5. Remove the chosen feature from the children's list (it cannot be used again on that path) and go back to step 2 for each child.

This makes a **multi-way tree**: a feature with five values gives five branches.

## A small worked example

Six mushrooms: 3 edible (e) and 3 poisonous (p). Two features, A and B, split them like this:

- Feature A: value 1 has 3 rows (2 e, 1 p), value 2 has 3 rows (1 e, 2 p)
- Feature B: value 1 has 2 rows (2 e, 0 p), value 2 has 4 rows (1 e, 3 p)

The same six rows are used in the READMEs of [gain-ratio](../gain-ratio/README.md), [gini-index](../gini-index/README.md) and [misclassification-error](../misclassification-error/README.md), so you can compare the rules.

The root: 3 e and 3 p, so $H(S) = 1$.

**Feature A**

- Each group is 2 against 1: $H = -\tfrac23 \log_2 \tfrac23 - \tfrac13 \log_2 \tfrac13 = 0.918$
- $H(S \mid A) = \tfrac36 (0.918) + \tfrac36 (0.918) = 0.918$
- $IG(S, A) = 1 - 0.918 = 0.082$

**Feature B**

- Value 1 is pure: $H = 0$
- Value 2 is 1 against 3: $H = -\tfrac14 \log_2 \tfrac14 - \tfrac34 \log_2 \tfrac34 = 0.811$
- $H(S \mid B) = \tfrac26 (0) + \tfrac46 (0.811) = 0.541$
- $IG(S, B) = 1 - 0.541 = 0.459$

Feature B wins by far. The "value 1" branch is already pure (leaf: edible), and the "value 2" branch continues with the remaining features.

## How the code works

1. **Load.** Read `../datasets/mushrooms.csv`. The `?` marks are replaced by the most common value of the column. Target: edible = 0, poisonous = 1. Split 80/20 with `random_state=42`. No encoding is needed, because ID3 works directly with category values.
2. **Store the tree as lists.** Each node has its rows, the features still available, its depth, its parent, the feature it splits on, its children (a dictionary from feature value to child number) and its class label. A pointer `n` walks through the nodes one by one (a queue), so parents are always handled before their children.
3. **For each node:** count the classes, set the label to the majority class, compute the entropy, stop if it is 0 or there are no features left. Otherwise compute the information gain of every available feature (and print it), choose the best, and create the children.
4. **Rules.** For every leaf, walk back up to the root to build a rule like `if odor = n and spore-print-color = k then edible`, and print it.
5. **Predict.** A test row starts at the root and follows the branch for its feature value. If it meets a value that was never seen at that node in training, it stops there and uses that node's majority class.
6. **Report** accuracy, precision, recall, F1, the error rate and the confusion matrix.

[id3_3.py](id3_3.py) does the same thing but loops over three criteria: `gain_ratio`, `gini` and `misclassification`. It prints the result and the number of leaves for each, so you can compare the trees.

## Library version

[id3tree.py](id3tree.py) uses `DecisionTreeClassifier(criterion="entropy")`. The scoring is the same, but scikit-learn builds **binary** trees: each node asks a yes-or-no question like "feature <= value". So the categories are first turned into numbers, and the tree shape can be different from the multi-way ID3 tree. The script prints the tree with `export_text`, plus the depth and number of leaves.

## Strengths and weaknesses

- Easy to read: the tree is a set of if-then rules anyone can follow.
- Needs no scaling of features.
- If it grows without limit, it can memorise noise in the training data (overfitting). Stopping early or pruning helps.
- Information gain likes features with many different values. [Gain ratio](../gain-ratio/README.md) fixes that.

## Things to try

- Look at the first rules printed and check if they make sense for mushrooms.
- Compare the number of leaves from `id3_3.py` for the three criteria.
