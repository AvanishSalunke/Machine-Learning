# Machine Learning

This repository collects common machine learning algorithms and their code. Most algorithms come in two versions:

- the main implementation, which follows the algorithm step by step with numpy
- the library version, which does the same job with a library such as scikit-learn, Keras, XGBoost or hmmlearn, and has a `lib` prefix on the file name

Every algorithm has a document in the [docs](docs) folder. It explains the idea, the formulas, and exactly what the code in this repository does, including the settings used.

## Regression and Classification Basics

Linear regression predicts a continuous value by fitting a weighted sum of the features. Logistic regression adds a sigmoid on top of it to predict class probabilities. Naive Bayes classifies with Bayes' theorem and the assumption that features are independent given the class.

- Linear regression: [linearregression.py](linearregression.py), [liblinearregression.py](liblinearregression.py), [notes](docs/linear-regression.md)
- Logistic regression: [logisticregression.py](logisticregression.py), [liblogisticregression.py](liblogisticregression.py), [notes](docs/logistic-regression.md)
- Naive Bayes: [naivebayes.py](naivebayes.py), [libnaivebayes.py](libnaivebayes.py), [notes](docs/naive-bayes.md)

## Decision Trees

A decision tree splits the data again and again on the feature that best separates the classes. The algorithms differ in how the best split is chosen.

- ID3 (information gain): [id3.py](id3.py), [id3_3.py](id3_3.py), library version [id3tree.py](id3tree.py), [notes](docs/id3.md)
- Gain ratio: [gainratio.py](gainratio.py), [notes](docs/gain-ratio.md)
- Gini index: [giniindex.py](giniindex.py), [notes](docs/gini-index.md)
- Misclassification error: [misclassificationerror.py](misclassificationerror.py), [notes](docs/misclassification-error.md)

## Support Vector Machines

An SVM separates two classes with the widest possible margin. Kernels let it draw curved boundaries.

- Linear SVM: [libsvmlinear.py](libsvmlinear.py), [notes](docs/svm-linear.md)
- Polynomial kernel: [svmpolynomial.py](svmpolynomial.py), [notes](docs/svm-polynomial.md)
- Gaussian (RBF) kernel: [svmkernel.py](svmkernel.py), [libsvmkernel.py](libsvmkernel.py), [notes](docs/svm-kernel.md)

## Ensemble Methods

Ensembles combine many simple models into one strong model.

- Bagging, which trains trees on bootstrap samples and takes a vote: [bagging.py](bagging.py), [libbagging.py](libbagging.py), [notes](docs/bagging.md)
- Random forest, which is bagging plus a random subset of features at each split: [randomforest.py](randomforest.py), [librandomforest.py](librandomforest.py), [notes](docs/random-forest.md)
- AdaBoost, which re-weights the samples that were classified wrongly: [adaboost.py](adaboost.py), [libadaboost.py](libadaboost.py), [notes](docs/adaboost.md)
- Gradient boosting, which fits each new tree to the errors of the current model: [gradientboosting.py](gradientboosting.py), [libgradientboosting.py](libgradientboosting.py), [notes](docs/gradient-boosting.md)
- XGBoost, a regularised gradient boosting that uses second order information: [xgboost.py](xgboost.py), [libxgboost.py](libxgboost.py), [notes](docs/xgboost.md)

## Neural Networks

- Feed-forward neural network, trained with backpropagation: [nn.py](nn.py), [libnn.py](libnn.py), [notes](docs/neural-network.md)
- RNN, a network with a hidden state that carries over from step to step: [rnn.py](rnn.py), [librnn.py](librnn.py), [notes](docs/rnn.md)
- LSTM, an RNN with gates and a cell state for long sequences: [lstm.py](lstm.py), [liblstm.py](liblstm.py), [notes](docs/lstm.md)
- GRU, a lighter gated RNN: [gru.py](gru.py), [libgru.py](libgru.py), [notes](docs/gru.md)

## Hidden Markov Models

An HMM models a system that moves between hidden states and only shows observations. Each of the four problems below has a plain version (small example), a dataset version (parameters estimated from data, with a `dataset` prefix) and a library version.

- Forward-backward, the probability of an observation sequence: [hmmfb.py](hmmfb.py), [datasethmmfb.py](datasethmmfb.py), [libhmmfb.py](libhmmfb.py), [notes](docs/hmm-forward-backward.md)
- Viterbi, the most likely hidden state sequence: [hmmviterbi.py](hmmviterbi.py), [datasethmmviterbi.py](datasethmmviterbi.py), [libhmmviterbi.py](libhmmviterbi.py), [notes](docs/hmm-viterbi.md)
- Baum-Welch, learning the model parameters from data: [hmmbaumwelch.py](hmmbaumwelch.py), [datasethmmbaumwelch.py](datasethmmbaumwelch.py), [libhmmbaumwelch.py](libhmmbaumwelch.py), [notes](docs/hmm-baum-welch.md)
- HMM classifier, one HMM per class: [hmmclassifier.py](hmmclassifier.py), [datasethmmclassifier.py](datasethmmclassifier.py), [libhmmclassifier.py](libhmmclassifier.py), [notes](docs/hmm-classifier.md)

## Preprocessing

[preprocessinggeneral.py](preprocessinggeneral.py) is a general template for cleaning, encoding, splitting and scaling a dataset. See [notes](docs/preprocessing.md).

## Datasets

- [electricity_cost_dataset.csv](electricity_cost_dataset.csv), used for linear regression
- [heart_cleveland_upload.csv](heart_cleveland_upload.csv), used for the SVMs
- [mushrooms.csv](mushrooms.csv), used for most of the classification algorithms

The `dataset` prefixed HMM scripts read a file named `dataset.csv`, which is not included in the repository.
