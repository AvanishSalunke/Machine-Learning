# Random Forest

Files: [randomforest.py](../randomforest.py) and [librandomforest.py](../librandomforest.py)

## Idea

A random forest is [bagging](bagging.md) of decision trees with one extra source of randomness. In bagging, all trees tend to choose the same strong features near the root, so they end up similar and their errors are correlated. A random forest limits each split to a random subset of the features, which makes the trees more different from each other. Averaging less correlated trees reduces variance more.

## The two kinds of randomness

1. **Bootstrap sampling of rows**: each tree is trained on $m$ rows drawn with replacement.
2. **Random feature subset at every split**: at each node, only $p$ randomly chosen features among the remaining ones are considered, and the best of those (highest information gain) is used. A common choice for classification is

$$p = \lfloor \sqrt{n} \rfloor$$

where $n$ is the total number of features.

## Prediction and OOB error

Prediction is by majority vote of the trees:

$$\hat{y}(\mathbf{x}) = \text{majority vote}\big\{ h_1(\mathbf{x}), \dots, h_T(\mathbf{x}) \big\}$$

As in bagging, rows left out of a tree's bootstrap sample are used to compute the out-of-bag error, an estimate of the generalisation error that needs no separate validation data.

## Split criterion

Entropy and information gain, as in [ID3](id3.md):

$$H(S) = -\sum_c p_c \log_2 p_c, \qquad IG(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|} H(S_v)$$

## What the code does

- Data: `mushrooms.csv` with `?` replaced by the mode, edible = 0 and poisonous = 1, 80/20 split with `random_state=42`.
- 10 trees, seed 42, $p = \lfloor \sqrt{\text{number of features}} \rfloor$ features tried at each node.
- Trees use multi-way splits on categorical features and each feature is used at most once on a path. A node becomes a leaf when it is pure or no features remain.
- OOB votes are collected per tree and the OOB error is printed. The test error is tracked as trees are added one by one and plotted.

## Library version

`librandomforest.py` uses scikit-learn's `RandomForestClassifier(n_estimators=10, criterion="entropy", max_features="sqrt", bootstrap=True, oob_score=True)`. Features are integer coded and the trees are binary, and `max_features` is applied at each split just like in the code version.
