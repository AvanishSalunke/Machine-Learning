# Data Preprocessing

Files in this folder:

- [preprocessinggeneral.py](preprocessinggeneral.py): a general template that cleans, encodes, splits and scales a dataset.

There is no library version, because the script already uses pandas and scikit-learn for each step.

## In simple words

Real data is messy. Some values are missing, some columns are text, and the numbers can be on very different scales: one column in thousands, another between 0 and 1. Most models either cannot read this directly or learn badly from it. Preprocessing is **cleaning the ingredients before cooking**. It is not glamorous, but it often affects the result more than the choice of model.

Most scripts in this repository repeat a short version of these steps at the top. This folder collects the full recipe in one place.

## Key words

- **Missing value**: an empty cell, or a placeholder such as `?`.
- **Imputation**: filling in missing values with a sensible guess.
- **One-hot encoding**: replacing a category column with several 0/1 columns, one per category.
- **Standardisation**: rescaling a column to have mean 0 and standard deviation 1.
- **Train/test split**: separating data used to learn from data used to check.
- **Data leakage**: information from the test set sneaking into training.

## The steps and the maths

**1. Look at the data.** Print the shape, the first rows and the number of missing values in every column. This shows what needs fixing.

**2. Handle missing values.**

- `?` is changed into a proper missing value (`NaN`).
- Duplicate rows are removed.
- A **numeric** column is filled with its **median** (the middle value). The median is not pulled around by extreme values, unlike the mean.
- A **text** column is filled with its **mode** (the most common value).

**3. Encode the target.** `LabelEncoder` turns class names into integers: for example `e` and `p` become 0 and 1.

**4. Encode the features.** Text columns are turned into 0/1 columns with `pd.get_dummies(drop_first=True)`. A column with $k$ categories becomes $k - 1$ columns. One category is dropped because it can be worked out from the others (all zeros means "the dropped one"), and keeping it would add a redundant column.

**5. Split into train and test.** The code keeps 80% for training and 20% for testing, with `random_state=42` so the split can be repeated. `stratify=y` keeps the same class proportions in both parts. Without it a small class could, by chance, be under-represented in the test set.

**6. Standardise the numbers.** For each column:

$$x' = \frac{x - \mu}{\sigma}, \qquad \mu = \frac{1}{m}\sum_i x_i, \qquad \sigma = \sqrt{\frac{1}{m}\sum_i (x_i - \mu)^2}$$

The mean $\mu$ and standard deviation $\sigma$ are computed on the **training** part only (`fit_transform`), and then the same values are used on the test part (`transform`). If the scaler also saw the test data, information about the test set would leak into training, and the test score would look better than it really is.

## A small worked example

A column with the training values $2, 4, 6$.

- $\mu = 4$
- $\sigma = \sqrt{\tfrac{(2-4)^2 + (4-4)^2 + (6-4)^2}{3}} = \sqrt{8/3} = 1.633$
- Scaled values: $\dfrac{2-4}{1.633} = -1.225$, $0$, and $+1.225$

A test value of $5$ is scaled with the **training** numbers: $(5 - 4)/1.633 = 0.612$.

One-hot example: a column `colour` with values `red`, `green`, `blue` becomes two columns after `drop_first=True`, for instance `colour_green` and `colour_red`. A `blue` row has 0 in both.

## How the code works

1. Load `../datasets/mushrooms.csv` and set the target column name (`class`).
2. Print the shape, the head and the missing-value counts.
3. Replace `?` by `NaN`, drop duplicates, and fill missing values with the median (numbers) or the mode (text).
4. Encode the target with `LabelEncoder`, and the features with `get_dummies(drop_first=True)`.
5. Split with `train_test_split(test_size=0.2, random_state=42, stratify=y)`.
6. Scale with `StandardScaler`: fit on the training part, apply to both parts.
7. Print the final shapes and the class counts.

## Why it matters for each model

- **Gradient-based models** (linear and logistic regression, neural networks, SVMs) train faster and more stably when the features have similar scales.
- **Distance and kernel models** (the Gaussian SVM) depend directly on the scale of each feature.
- **Trees and ensembles** (ID3, random forest, boosting) do not need scaling, but it does no harm.

## Things to try

- Remove `stratify=y` and look at the class counts in the split.
- Fit the scaler on the full data (the wrong way) and see how the numbers differ.
