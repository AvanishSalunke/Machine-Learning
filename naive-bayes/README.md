# Naive Bayes

Files: [naivebayes.py](../naivebayes.py) and [libnaivebayes.py](../libnaivebayes.py)

## Idea

Naive Bayes is a probabilistic classifier built on Bayes' theorem. For a sample with features $x_1, \dots, x_n$ it asks which class $c$ is the most probable.

$$P(c \mid x_1, \dots, x_n) = \frac{P(c) \, P(x_1, \dots, x_n \mid c)}{P(x_1, \dots, x_n)}$$

The "naive" assumption is that the features are independent of each other once the class is known:

$$P(x_1, \dots, x_n \mid c) = \prod_{j=1}^{n} P(x_j \mid c)$$

The denominator is the same for every class, so it can be ignored when only comparing classes:

$$\hat{c} = \arg\max_c \; P(c) \prod_{j=1}^{n} P(x_j \mid c)$$

## Estimating the probabilities

All the mushroom features are categorical, so the probabilities come from simple counting.

Prior of a class, from the training set:

$$P(c) = \frac{N_c}{N}$$

Likelihood with Laplace (add-one) smoothing:

$$P(x_j = v \mid c) = \frac{n_{c,j,v} + 1}{N_c + k_j}$$

Here $n_{c,j,v}$ is how many training rows of class $c$ have value $v$ in feature $j$, $N_c$ is the number of rows of class $c$, and $k_j$ is the number of different values feature $j$ can take. The +1 makes sure a value never seen with a class does not get probability zero, which would wipe out the whole product.

## What the code does

- Data: `mushrooms.csv`. Missing values marked with `?` are replaced with the most frequent value of the column (the mode).
- Target: edible = 0, poisonous = 1. 80/20 train and test split with `random_state=42`.
- The implementation builds a dictionary of every conditional probability $P(x_j = v \mid c)$, prints them, then multiplies the prior with the likelihoods for each test row and takes the class with the larger score. The scores for the first three test rows are printed.
- Multiplying many small numbers can underflow on larger problems. A common fix is to add logarithms instead of multiplying: $\log P(c) + \sum_j \log P(x_j \mid c)$.

## Library version

`libnaivebayes.py` uses scikit-learn's `CategoricalNB(alpha=1)`, which is the same model: categorical features with Laplace smoothing where `alpha=1` is the add-one. The categories are first converted to integer codes. It works with log probabilities internally. `min_categories` makes sure that every category seen anywhere in the data is known, as in the manual version.
