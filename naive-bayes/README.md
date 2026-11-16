# Naive Bayes

Files in this folder:

- [naivebayes.py](naivebayes.py): probabilities counted and multiplied step by step with numpy
- [libnaivebayes.py](libnaivebayes.py): the same with scikit-learn's `CategoricalNB`

## In simple words

Suppose a doctor sees a patient with a fever and a cough and wants to know which illness is more likely. The doctor thinks: "How common is each illness? And among people with that illness, how often do I see a fever? A cough?" Naive Bayes does exactly this for every class and picks the class with the biggest answer.

It is called **naive** because it treats the symptoms as unrelated to each other once the illness is known. That is rarely exactly true, but it makes the maths very easy and the method works surprisingly well.

## Key words

- **Prior** $P(c)$: how common the class is before looking at any feature.
- **Likelihood** $P(x_j \mid c)$: how often a feature value shows up inside one class.
- **Posterior** $P(c \mid \mathbf{x})$: the probability of the class after seeing the features. This is what we want.
- **Laplace smoothing**: adding 1 to every count so that nothing has a probability of exactly zero.

## The maths, step by step

**1. Bayes' theorem.** It turns "how often do the features appear in this class" into "how likely is this class given the features":

$$P(c \mid x_1, \dots, x_n) = \frac{P(c)\; P(x_1, \dots, x_n \mid c)}{P(x_1, \dots, x_n)}$$

**2. The naive assumption.** Features are independent given the class, so the joint probability splits into a product:

$$P(x_1, \dots, x_n \mid c) = \prod_{j=1}^{n} P(x_j \mid c)$$

**3. The decision rule.** The bottom part, $P(x_1, \dots, x_n)$, is the same for every class, so we skip it when comparing:

$$\hat{c} = \arg\max_{c} \; P(c) \prod_{j=1}^{n} P(x_j \mid c)$$

($\arg\max$ means "the class that gives the largest value".)

**4. The prior** is counted from the training rows, where $N_c$ rows have class $c$ out of $N$ in total:

$$P(c) = \frac{N_c}{N}$$

**5. The likelihood with Laplace smoothing.** Let $n_{c,j,v}$ be the number of rows of class $c$ whose feature $j$ equals value $v$, and let $k_j$ be the number of different values feature $j$ can take:

$$P(x_j = v \mid c) = \frac{n_{c,j,v} + 1}{N_c + k_j}$$

Without the "+1", a value that never appeared with a class in training would get probability 0, and since we multiply, one zero would wipe out the whole product. Adding 1 to the top and $k_j$ to the bottom keeps the probabilities of all values of a feature adding up to exactly 1.

**6. A note about underflow.** Multiplying 22 small numbers gives a very small number. On bigger problems it could round to zero, so it is common to add logarithms instead: $\log P(c) + \sum_j \log P(x_j \mid c)$. The class with the biggest sum is the same class as before.

## A small worked example

Training data for one feature, *odor*, with values `none` or `foul`:

- Edible: 3 rows with odor `none`, `none`, `foul`.
- Poisonous: 2 rows with odor `foul`, `foul`.

Priors: $P(e) = 3/5 = 0.6$ and $P(p) = 2/5 = 0.4$. The feature has $k = 2$ values.

- $P(\text{none} \mid e) = (2+1)/(3+2) = 0.6$ and $P(\text{foul} \mid e) = (1+1)/(3+2) = 0.4$
- $P(\text{none} \mid p) = (0+1)/(2+2) = 0.25$ and $P(\text{foul} \mid p) = (2+1)/(2+2) = 0.75$

A new mushroom smells `foul`:

- Edible: $0.6 \times 0.4 = 0.24$
- Poisonous: $0.4 \times 0.75 = 0.30$

The poisonous score is larger, so the prediction is **poisonous**. Notice that `none` never appeared with the poisonous class, yet thanks to smoothing it still gets 0.25 instead of 0.

## How the code works

1. **Load.** Read `../datasets/mushrooms.csv`. The `?` marks (missing values) are replaced by the most common value of that column.
2. **Split** 80/20 with `random_state=42`. The target is edible = 0, poisonous = 1.
3. **Count.** For every feature, every value and every class, the code computes the smoothed likelihood from the formula above and stores it in a dictionary. They are also printed, so you can read the model.
4. **Predict.** For each test row, start with the prior of a class, multiply by the likelihood of each of its feature values, and keep the class with the larger product. The scores of the first three test rows are printed.
5. **Report** accuracy, precision, recall, F1, the error rate and the confusion matrix.

There is no training loop, because everything comes from counting.

## Library version

`libnaivebayes.py` uses `CategoricalNB(alpha=1)`. It is the same model: counts with add-one smoothing (`alpha=1`). The text categories are first turned into integer codes. Inside, it adds logarithms to avoid underflow. `min_categories` tells it how many values each feature can take, so that a value seen only in the test set does not break it.

## Strengths and weaknesses

- Very fast, needs little data, easy to understand, and the probabilities can be read directly.
- The independence assumption is often wrong, which can make the probabilities over-confident, even when the predicted class is still right.
- Works best with categorical data like this dataset.

## Things to try

- Change the smoothing from 1 to 0.1 or to 5 and see what happens.
- Print which feature values have the biggest ratio between the two classes. These are the strongest clues.
