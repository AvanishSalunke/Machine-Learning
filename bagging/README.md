# Bagging

Files in this folder:

- [bagging.py](bagging.py): bagging of ID3 trees, step by step with numpy.
- [libbagging.py](libbagging.py): library version with scikit-learn's `BaggingClassifier`.

## In simple words

Ask one person to guess the number of sweets in a jar and the answer may be far off. Ask a hundred people and average the guesses, and the answer is usually very good, because the random errors cancel out.

Bagging does this with decision trees. A single deep tree is very sensitive: change a few training rows and you get a different tree. So we train **many** trees, each on a slightly different random version of the training data, and let them **vote**. The name comes from **b**ootstrap **agg**regat**ing**.

## Key words

- **Bootstrap sample**: a new training set made by drawing rows from the original **with replacement**, so some rows appear twice or more and some not at all.
- **Ensemble**: a group of models that decide together.
- **Variance**: how much a model changes when the training data changes. Bagging reduces it.
- **Out-of-bag (OOB) rows**: the rows that a given tree did not see in its bootstrap sample.
- **Majority vote**: the class chosen by most trees.

## The maths, step by step

**1. Bootstrap sampling.** With $m$ training rows, draw $m$ times, each time picking a row at random and putting it back. The chance that a particular row is **never** picked in $m$ draws is

$$\left(1 - \frac{1}{m}\right)^m \;\xrightarrow{m \to \infty}\; e^{-1} \approx 0.368$$

So each tree sees about 63% different rows, and about 37% are left out. These left-out rows are that tree's OOB rows. (For $m = 10$ the number is $0.9^{10} = 0.35$ already.)

**2. Train $T$ trees**, one on each bootstrap sample. In this code each tree is a full ID3 tree (see [ID3](../id3/README.md)), grown until its nodes are pure.

**3. Combine by majority vote:**

$$\hat{y}(\mathbf{x}) = \text{the class chosen most often among } h_1(\mathbf{x}), \dots, h_T(\mathbf{x})$$

**4. Why does voting help?** Suppose each tree has variance $\sigma^2$ and the trees were fully independent. Averaging $T$ of them cuts the variance to $\sigma^2 / T$. The trees are not fully independent (they share most data), so the gain is smaller, but still clear. If the trees have pairwise correlation $\rho$, the variance of the average is

$$\rho\, \sigma^2 + \frac{1 - \rho}{T}\, \sigma^2$$

That is why making the trees **less correlated** helps even more. [Random forest](../random-forest/README.md) does exactly that.

**5. The OOB error.** Each training row is out-of-bag for roughly a third of the trees. Let only those trees vote on it. These trees have never seen the row, so this is a fair test. The fraction of wrong OOB votes is the OOB error, a free estimate of the test error that needs no extra data.

## A small worked example

Ten trees vote on one mushroom: 7 say "poisonous" and 3 say "edible". The majority wins, so the answer is poisonous.

The 7-against-3 split also tells you something: it is not unanimous, so the sample is somewhat uncertain. The fraction of votes (0.7) is a rough probability.

## How the code works

1. **Load** `../datasets/mushrooms.csv` and replace `?` with the most common value of the column. Edible = 0, poisonous = 1. **Split** 80/20 with `random_state=42`. The trees work directly on category values.
2. **For each of 10 trees** (random seed 42):
   - Draw a bootstrap sample of the training rows. The OOB rows are the ones not drawn.
   - Grow a full ID3 tree on the sample: at each node, choose the feature with the highest information gain, make one child per value, stop at pure nodes or when no features remain.
   - Run the OOB rows through the tree and record its votes for them.
3. **OOB error.** For each training row that was out-of-bag at least once, the class with the most OOB votes is its prediction. The fraction of wrong ones is printed.
4. **Test.** Each test row goes through every tree. The votes are added one tree at a time, so the code can compute the test error after 1 tree, 2 trees, and so on up to 10. That curve is plotted, and the final vote gives the predictions.
5. **Report** accuracy, precision, recall, F1 and the confusion matrix.

## Library version

[libbagging.py](libbagging.py) uses `BaggingClassifier(DecisionTreeClassifier(criterion="entropy"), n_estimators=10, bootstrap=True, oob_score=True)`. The idea is the same. The categories are first turned into numbers and scikit-learn grows binary trees. The OOB error comes from `1 - oob_score_`, and the per-tree test error curve is built by adding up the trees' votes one by one.

## Strengths and weaknesses

- Easy to use, hard to break, and reduces overfitting of deep trees.
- The trees are still similar to each other, because all of them can pick the same strong features. [Random forest](../random-forest/README.md) improves this.
- Less readable than a single tree.

## Things to try

- Change the number of trees to 1, 5, 50 and look at the test error curve.
- Compare the OOB error with the test error. They should be close.
