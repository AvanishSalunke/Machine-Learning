# K-Fold Cross-Validation

Files in this folder:

- [kfold.py](kfold.py): K-fold cross-validation written step by step with numpy (the folds are made by hand).
- [libkfold.py](libkfold.py): the same task with scikit-learn's `KFold` and `cross_validate`.

## In simple words

Imagine a teacher who wants to know how well a student really knows the subject. If the teacher asks only **one** set of questions, the result may be luck: the questions might happen to be easy, or hard. A fairer way is to prepare five different sets of questions, test the student on each, and look at the average.

Evaluating a model has the same problem. The usual single split (80% train, 20% test) gives one score, and that score depends on which rows happened to land in the test set. **K-fold cross-validation** repeats the test $K$ times:

1. Cut the data into $K$ equal parts (the **folds**).
2. Use fold 1 as the test set and the other folds as the training set. Record the score.
3. Use fold 2 as the test set, the rest for training. Record the score.
4. Continue until every fold has been the test set once.
5. Report the **mean** of the $K$ scores, and how much they vary.

Every sample is used for testing exactly once, and for training $K - 1$ times, so no data is wasted.

## Key words

- **Fold**: one of the $K$ parts of the data.
- **Cross-validation score**: the average of the scores on the $K$ test folds.
- **Data leakage**: information from the test fold sneaking into training. It makes the score look better than it is.
- **Pipeline**: a chain of steps (here scaling, then the model) that is fitted again inside every fold.

## The maths, step by step

With $m$ samples and $K$ folds:

**1. Shuffle the sample numbers** with a fixed seed, so the folds are random but repeatable.

**2. Fold sizes.** If $m$ is not a multiple of $K$, the first $m \bmod K$ folds get one sample more:

$$|F_i| = \left\lfloor \frac{m}{K} \right\rfloor + \mathbb{1}\big[i \le m \bmod K\big]$$

**3. For each fold $i = 1, \dots, K$:** the test set is $F_i$ and the training set is everything else, $\bigcup_{j \ne i} F_j$. Fit the scaler **on the training part only**, scale both parts with it, fit the model on the training part, and compute the score $s_i$ on the test part.

**4. The final result** is the mean and the spread of the fold scores:

$$\bar{s} = \frac{1}{K} \sum_{i=1}^{K} s_i, \qquad \text{std} = \sqrt{\frac{1}{K} \sum_{i=1}^{K} (s_i - \bar{s})^2}$$

The mean is the estimate of how well the model will do on new data. The standard deviation tells you how much to trust it: a large value means the score depends a lot on the data split.

**5. Choosing $K$.**

- Small $K$ (like 2 or 3): fast, but each model trains on less data, so the score is pessimistic.
- $K = 5$ or $K = 10$ are the usual choices, a good balance.
- $K = m$ is called **leave-one-out**: each sample is a test set of its own. It uses the most data but is slow.

**6. Why the scaler is fitted inside every fold.** If you scale the whole dataset first, the mean and standard deviation used for scaling include the test fold, so a little information about it reaches training. That is leakage. Doing the scaling inside each fold keeps the test fold truly unseen.

## A small worked example

Twenty samples and $K = 4$ folds, so every fold has 5 samples. The model is trained 4 times:

- Fold 1 is the test set, accuracy 0.8 (4 of 5 right)
- Fold 2 is the test set, accuracy 1.0
- Fold 3 is the test set, accuracy 0.6
- Fold 4 is the test set, accuracy 0.8

Mean: $(0.8 + 1.0 + 0.6 + 0.8) / 4 = 0.8$. Standard deviation: $\sqrt{(0 + 0.04 + 0.04 + 0)/4} = 0.14$.

A single split might have given 1.0 or 0.6 by luck. The cross-validation score $0.80 \pm 0.14$ is more honest.

## How the code works

1. **Load** `../datasets/heart_cleveland_upload.csv`. The target `condition` is 0 or 1. All 297 rows are used, because the folds replace the single train/test split.
2. **Make the folds.** `np.arange(rows)` gives the row numbers, and `np.random.RandomState(42).shuffle` mixes them. The fold sizes are computed with the formula above. With 297 rows and 5 folds, the sizes are 60, 60, 59, 59, 59.
3. **Loop over the folds.** For fold $i$, the test rows are a slice of the shuffled numbers, and the training rows are all the others. The code scales the features with a `StandardScaler` fitted on the training rows only, trains a logistic regression (see [logistic regression](../logistic-regression/README.md)), predicts the test rows, and stores the accuracy, precision, recall and F1.
4. **Report** the accuracy of every fold, and the mean and standard deviation of the accuracy, plus the mean precision, recall and F1. A bar chart shows the five accuracies with a red line at the mean.

Settings: 5 folds, seed 42, logistic regression with `max_iter=1000`.

What it finds on this data: the fold accuracies are 0.733, 0.900, 0.864, 0.881 and 0.814. The mean is **0.839** with a standard deviation of about 0.060. The lowest and highest folds differ by almost 17 points, which shows how misleading one split can be. If you had taken only the first fold you would have reported 73% accuracy, and only the second would have given 90%.

The model here is logistic regression, but the idea works with any model: swap the model line for another classifier and the loop stays the same.

## Library version

[libkfold.py](libkfold.py) uses three scikit-learn pieces:

- `KFold(n_splits=5, shuffle=True, random_state=42)` makes the folds. Internally it shuffles the row numbers with the same random generator and cuts them the same way, so the folds are **identical** to the ones made by hand.
- `make_pipeline(StandardScaler(), LogisticRegression(...))` joins the scaler and the model, so the scaler is refitted inside every fold automatically.
- `cross_validate(...)` runs the whole loop and returns the scores for accuracy, precision, recall and F1.

On this data both files print exactly the same fold sizes and the same scores.

## Strengths and weaknesses

- A far more reliable score than one split, and it uses all the data for both training and testing.
- It costs $K$ times the training time.
- For classes with very different sizes, plain K-fold can give a fold with almost no samples of the small class. Stratified K-fold (`StratifiedKFold`) keeps the class proportions in every fold.
- Not suitable as it is for data where the order matters, such as time series.

## Things to try

- Change `folds` to 3, 10 and to the number of rows (leave-one-out) and watch the mean and the standard deviation.
- Replace the model with [kNN](../knn/README.md) and compare the cross-validation accuracy of the two.
- Use `StratifiedKFold` in the library version and compare.
