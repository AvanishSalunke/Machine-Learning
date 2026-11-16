# Data Preprocessing

File: [preprocessinggeneral.py](preprocessinggeneral.py)

A general template for preparing a tabular dataset before training a model. Most scripts in this repository repeat a trimmed version of these steps at the top.

## Steps

1. **Inspect the data.** Print the shape, the first rows and the number of missing values per column.
2. **Handle missing values.** Placeholders such as `?` are turned into proper missing values. Duplicate rows are dropped. Numeric columns are filled with the median, which is not thrown off by outliers. Categorical columns are filled with the mode (most frequent value).
3. **Encode the target.** `LabelEncoder` turns the class labels into the integers 0, 1, 2, ...
4. **Encode the features.** Categorical features are one-hot encoded with `pd.get_dummies(drop_first=True)`. One dummy column per category is dropped to avoid redundant columns.
5. **Split.** 80% train and 20% test with `random_state=42`. `stratify=y` keeps the class proportions the same in both parts.
6. **Scale.** Features are standardised with `StandardScaler`:

$$x' = \frac{x - \mu}{\sigma}$$

The mean $\mu$ and standard deviation $\sigma$ are computed on the training data only and then applied to the test data. Fitting the scaler on the full data would leak information from the test set into training.

## Why it matters

- Gradient based models (linear and logistic regression, SVM, neural networks) train faster and more stably when the features are on the same scale.
- Distance and kernel based models (the Gaussian SVM) are directly affected by the scale of each feature.
- Tree based models (ID3, random forest, boosting) do not need scaling, but it does no harm.
- Mean and median imputation use only values from the data, so each choice should be made with the training data only when the results are going to be reported honestly.
