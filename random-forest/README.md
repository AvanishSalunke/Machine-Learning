# Random Forest

Files in this folder:

- [randomforest.py](randomforest.py): a random forest of ID3 trees, step by step with numpy.
- [librandomforest.py](librandomforest.py): library version with scikit-learn's `RandomForestClassifier`.

## In simple words

[Bagging](../bagging/README.md) trains many trees on different random samples of the rows. But there is a catch: if one feature is very strong (say, the smell of the mushroom), almost every tree will use it at the top, so all the trees end up looking alike, and their mistakes are alike too. Voting only helps when the voters disagree in useful ways.

A random forest adds a second kind of randomness to force variety. At **every split**, a tree is only allowed to look at a **small random group of the features**, and it picks the best one inside that group. Sometimes the strong feature is not in the group, so the tree has to find another good way to split. The trees become more different, and the vote becomes stronger.

## Key words

- **Bootstrap sample**: rows drawn with replacement for each tree.
- **Feature subset** (size $p$): the random group of features a node may choose from.
- **Out-of-bag (OOB) error**: error measured on the rows a tree did not train on.
- **Decorrelated trees**: trees whose mistakes are not the same mistakes.

## The maths, step by step

**1. Two kinds of randomness for each tree.**

- Rows: draw $m$ rows with replacement (a bootstrap sample), as in [bagging](../bagging/README.md).
- Features: at each node, pick $p$ features at random from the ones still available. A common choice for classification is

$$p = \lfloor \sqrt{n} \rfloor$$

with $n$ the total number of features. For the mushroom data, $n = 22$, so $p = 4$.

**2. The split rule.** Among the $p$ chosen features, use the one with the highest information gain, exactly as in [ID3](../id3/README.md):

$$H(S) = -\sum_c p_c \log_2 p_c, \qquad IG(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|} H(S_v)$$

**3. The vote.**

$$\hat{y}(\mathbf{x}) = \text{the class chosen most often among } h_1(\mathbf{x}), \dots, h_T(\mathbf{x})$$

**4. Why it works.** For trees with variance $\sigma^2$ and correlation $\rho$ between them, the variance of the average is

$$\rho\, \sigma^2 + \frac{1 - \rho}{T}\, \sigma^2$$

Adding more trees shrinks the second part. Picking random features lowers $\rho$, which shrinks the first part. Bagging cannot do that.

**5. OOB error.** For every training row, take the votes of only those trees that did not see it in their bootstrap sample. The fraction of rows predicted wrongly is the OOB error, a built-in test error estimate.

## A small worked example

A mushroom dataset has 22 features, so $p = \lfloor \sqrt{22} \rfloor = 4$. At the root, tree 1 gets the random features {cap-shape, gill-size, habitat, ring-type}. It cannot use `odor` here, even though `odor` is the strongest feature overall, so it picks the best among the four, maybe `gill-size`. Tree 2 gets {odor, veil-color, stalk-shape, population} and picks `odor`. The two trees now look quite different, but both are fairly good, and their votes complement each other.

## How the code works

1. **Load** `../datasets/mushrooms.csv`, replace `?` with the most common value, edible = 0, poisonous = 1. **Split** 80/20 with `random_state=42`.
2. **For each of 10 trees** (seed 42):
   - Draw a bootstrap sample of rows; the rows not drawn are the OOB rows.
   - Grow the tree node by node. At each node: stop if the node is pure or no features remain. Otherwise choose $p = \lfloor\sqrt{n}\rfloor$ features at random from the remaining ones, compute the information gain of each, split on the best, and make one child per value.
   - Run the OOB rows through the tree and store the votes.
3. **OOB error** is printed from the stored votes.
4. **Test.** Each test row goes through the trees. The votes are added one tree at a time, which gives the test error after 1, 2, ..., 10 trees (plotted). The final vote is the prediction.
5. **Report** accuracy, precision, recall, F1 and the confusion matrix.

## Library version

[librandomforest.py](librandomforest.py) uses `RandomForestClassifier(n_estimators=10, criterion="entropy", max_features="sqrt", bootstrap=True, oob_score=True)`. The same two sources of randomness are used. The categories are first turned into numbers and the trees are binary. The OOB error is `1 - oob_score_`, and the error-versus-trees curve is computed in the same way as in the numpy version.

## Strengths and weaknesses

- Strong results with almost no tuning, handles many features, and the OOB error gives a free estimate of test error.
- Less readable than a single tree, and slower to train and predict.
- The more trees, the better, up to a point where the error stops improving.

## Things to try

- Compare the test error curve with the one from [bagging](../bagging/README.md). Which falls faster?
- Change $p$ to 1 or to all the features. With all features, this becomes bagging.
