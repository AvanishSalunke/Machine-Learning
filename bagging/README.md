# Bagging

Files: [bagging.py](bagging.py) and [libbagging.py](libbagging.py)

## Idea

Bagging stands for bootstrap aggregating. A single deep decision tree has low bias but high variance: a small change in the training data can change the tree a lot. Bagging trains many trees on different random versions of the data and lets them vote, which averages out the variance.

## Bootstrap sampling

For each of the $T$ trees, draw $m$ rows from the $m$ training rows **with replacement**. Some rows appear several times, others not at all. The chance that a given row is left out of one bootstrap sample is

$$\left(1 - \frac{1}{m}\right)^m \;\xrightarrow{m \to \infty}\; e^{-1} \approx 0.368$$

so about 37% of the rows are left out of each tree. These are the out-of-bag (OOB) rows of that tree.

## Aggregation

Each tree votes for a class and the majority wins:

$$\hat{y}(\mathbf{x}) = \text{majority vote}\big\{ h_1(\mathbf{x}), \dots, h_T(\mathbf{x}) \big\}$$

## Out-of-bag error

Every training row is out-of-bag for roughly a third of the trees. Collecting the votes only from those trees gives a prediction for that row made by models that never saw it. The fraction of wrong OOB predictions is the OOB error, a built-in estimate of test error without a separate validation set.

## What the code does

- Data: `mushrooms.csv` with `?` replaced by the column mode. Edible = 0, poisonous = 1. 80/20 split with `random_state=42`.
- 10 trees, random seed 42. Each tree is a full ID3 tree (entropy and information gain, multi-way splits, see [id3.md](../id3/README.md)) grown on its bootstrap sample until the nodes are pure or no features are left.
- After each tree is built, the OOB rows are run through it and the votes are stored. The OOB error is printed.
- On the test set the votes of the trees are accumulated one tree at a time, so the test error after 1, 2, ..., 10 trees can be plotted.

## Library version

`libbagging.py` uses scikit-learn's `BaggingClassifier` with `DecisionTreeClassifier(criterion="entropy")`, 10 estimators, `bootstrap=True` and `oob_score=True`. The categories are converted to integer codes first, and the trees are binary. Other behaviour (OOB error and test error per tree count) is the same.

The next step beyond bagging is [random forest](../random-forest/README.md), which also randomises the features.
