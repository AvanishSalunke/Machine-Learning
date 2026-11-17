# Machine Learning

A collection of machine learning algorithms with clean code, library versions to compare against, and a plain-language README for every algorithm. Each README explains the idea in simple words, shows every formula, works through a small example with numbers, and describes exactly what the code does.

## Contents

- [Quick start](#quick-start)
- [How the repository is organised](#how-the-repository-is-organised)
- [Algorithms](#algorithms)
- [Datasets](#datasets)
- [Contributing](#contributing)
- [License](#license)

## Quick start

You need Python 3.9 or newer and git.

1. Get the code:

```bash
git clone https://github.com/AvanishSalunke/Machine-Learning.git
cd Machine-Learning
```

2. Make a virtual environment, so the packages stay separate from the rest of your system (optional but recommended):

```bash
python -m venv venv
```

Activate it. On Windows:

```bash
venv\Scripts\activate
```

On Mac or Linux:

```bash
source venv/bin/activate
```

3. Install the packages:

```bash
pip install -r requirements.txt
```

4. Run an algorithm. Go into its folder first, because the scripts find the datasets with a relative path:

```bash
cd linear-regression
python linearregression.py
```

Many scripts draw a plot at the end, so a window may open. Close it to finish the script.

## How the repository is organised

Every algorithm has its own folder. Inside it you will find:

- the main code file, which follows the algorithm step by step with numpy
- the library version, with a `lib` prefix on the name (for example `liblinearregression.py`), which does the same job with scikit-learn, Keras, XGBoost or hmmlearn
- a `README.md` that teaches the algorithm

```
Machine-Learning/
  datasets/              the CSV files used by the scripts
  linear-regression/
    linearregression.py
    liblinearregression.py
    README.md
  id3/
    ...
  README.md              this file
  CONTRIBUTING.md        how to help
  LICENSE
  requirements.txt
```

A good way to study: read the README of an algorithm, run the main script, run the library version, and compare the two outputs.

## Algorithms

### Basics: regression and classification

- [Linear regression](linear-regression/README.md): fits a straight line or plane to predict a number, using gradient descent.
- [Logistic regression](logistic-regression/README.md): a linear model with a sigmoid on top that predicts the probability of a class.
- [Naive Bayes](naive-bayes/README.md): classifies with Bayes' theorem and counted probabilities.

### Decision trees

A decision tree asks a chain of questions about the features. The folders below differ in how the best question is chosen.

- [ID3](id3/README.md): picks the split with the biggest information gain (entropy). Includes a library version.
- [Gain ratio](gain-ratio/README.md): information gain divided by split information, so features with many values are not favoured.
- [Gini index](gini-index/README.md): picks the split with the lowest Gini impurity.
- [Misclassification error](misclassification-error/README.md): picks the split with the lowest error rate.

### Distance-based methods and clustering

- [kNN](knn/README.md): classifies a sample by a vote of its $k$ nearest training samples.
- [K-means](k-means/README.md): groups samples into $K$ clusters around moving centres, with no labels needed.

### Dimensionality reduction

- [PCA](pca/README.md): finds the directions of largest variance and keeps only the first few.

### Support vector machines

An SVM separates two classes with the widest possible margin.

- [Linear SVM](svm-linear/README.md): a straight boundary.
- [Polynomial kernel SVM](svm-polynomial/README.md): curved boundaries from polynomial features.
- [Gaussian kernel SVM](svm-kernel/README.md): flexible boundaries from the RBF kernel.

### Ensembles

Ensembles combine many simple models into a strong one.

- [Bagging](bagging/README.md): trees on bootstrap samples, combined by voting.
- [Random forest](random-forest/README.md): bagging plus a random choice of features at every split.
- [AdaBoost](adaboost/README.md): models trained in turn, each focusing on the samples the last one got wrong.
- [Gradient boosting](gradient-boosting/README.md): each new tree fixes the errors of the model so far.
- [XGBoost](xgboost/README.md): gradient boosting with second order information and regularisation.

### Neural networks

- [Neural network](neural-network/README.md): layers of neurons trained with backpropagation.
- [RNN](rnn/README.md): a network with memory for sequences.
- [LSTM](lstm/README.md): an RNN with gates that keep information for a long time.
- [GRU](gru/README.md): a lighter gated RNN.

### Hidden Markov models

An HMM has hidden states that we cannot see, and observations that we can. These folders also contain `dataset` versions that learn the model from data.

- [Forward-backward](hmm-forward-backward/README.md): how likely is an observation sequence?
- [Viterbi](hmm-viterbi/README.md): what is the most likely sequence of hidden states?
- [Baum-Welch](hmm-baum-welch/README.md): learn the model from observations.
- [HMM classifier](hmm-classifier/README.md): one HMM per class, to classify sequences.

### Model evaluation

- [K-fold cross-validation](k-fold-cross-validation/README.md): tests a model on $K$ different splits and averages the scores.

### Data preparation

- [Preprocessing](preprocessing/README.md): cleaning, encoding, splitting and scaling a dataset.

## Datasets

All datasets are in the [datasets](datasets) folder.

- [electricity_cost_dataset.csv](datasets/electricity_cost_dataset.csv): used by linear regression.
- [heart_cleveland_upload.csv](datasets/heart_cleveland_upload.csv): used by the SVMs.
- [mushrooms.csv](datasets/mushrooms.csv): used by most of the classification algorithms.

The `dataset` versions of the HMM scripts read a file called `dataset.csv`. It is not included, so add your own file with a `state` and an `obs` column (or a `class` column for the classifier) before running them.

## Contributing

Pull requests are welcome. The [contributing guide](CONTRIBUTING.md) explains how to fork, branch and open a pull request, and how a new algorithm folder should look.

## License

This project is released under the [MIT License](LICENSE).
