# Machine Learning

This repository collects common machine learning algorithms and their code. Most algorithms come in two versions:

- the main implementation, which follows the algorithm step by step with numpy
- the library version, which does the same job with a library such as scikit-learn, Keras, XGBoost or hmmlearn, and has a `lib` prefix on the file name

Every algorithm has its own folder with the code, the library version and a README. The README explains the idea, the formulas, and exactly what the code in this repository does, including the settings used.

## Regression and Classification Basics

Linear regression predicts a continuous value by fitting a weighted sum of the features. Logistic regression adds a sigmoid on top of it to predict class probabilities. Naive Bayes classifies with Bayes' theorem and the assumption that features are independent given the class.

- Linear regression: [linearregression.py](linear-regression/linearregression.py), [liblinearregression.py](linear-regression/liblinearregression.py), [notes](linear-regression/README.md)
- Logistic regression: [logisticregression.py](logistic-regression/logisticregression.py), [liblogisticregression.py](logistic-regression/liblogisticregression.py), [notes](logistic-regression/README.md)
- Naive Bayes: [naivebayes.py](naive-bayes/naivebayes.py), [libnaivebayes.py](naive-bayes/libnaivebayes.py), [notes](naive-bayes/README.md)

## Decision Trees

A decision tree splits the data again and again on the feature that best separates the classes. The algorithms differ in how the best split is chosen.

- ID3 (information gain): [id3.py](id3/id3.py), [id3_3.py](id3/id3_3.py), library version [id3tree.py](id3/id3tree.py), [notes](id3/README.md)
- Gain ratio: [gainratio.py](gain-ratio/gainratio.py), [notes](gain-ratio/README.md)
- Gini index: [giniindex.py](gini-index/giniindex.py), [notes](gini-index/README.md)
- Misclassification error: [misclassificationerror.py](misclassification-error/misclassificationerror.py), [notes](misclassification-error/README.md)

## Support Vector Machines

An SVM separates two classes with the widest possible margin. Kernels let it draw curved boundaries.

- Linear SVM: [libsvmlinear.py](svm-linear/libsvmlinear.py), [notes](svm-linear/README.md)
- Polynomial kernel: [svmpolynomial.py](svm-polynomial/svmpolynomial.py), [notes](svm-polynomial/README.md)
- Gaussian (RBF) kernel: [svmkernel.py](svm-kernel/svmkernel.py), [libsvmkernel.py](svm-kernel/libsvmkernel.py), [notes](svm-kernel/README.md)

## Ensemble Methods

Ensembles combine many simple models into one strong model.

- Bagging, which trains trees on bootstrap samples and takes a vote: [bagging.py](bagging/bagging.py), [libbagging.py](bagging/libbagging.py), [notes](bagging/README.md)
- Random forest, which is bagging plus a random subset of features at each split: [randomforest.py](random-forest/randomforest.py), [librandomforest.py](random-forest/librandomforest.py), [notes](random-forest/README.md)
- AdaBoost, which re-weights the samples that were classified wrongly: [adaboost.py](adaboost/adaboost.py), [libadaboost.py](adaboost/libadaboost.py), [notes](adaboost/README.md)
- Gradient boosting, which fits each new tree to the errors of the current model: [gradientboosting.py](gradient-boosting/gradientboosting.py), [libgradientboosting.py](gradient-boosting/libgradientboosting.py), [notes](gradient-boosting/README.md)
- XGBoost, a regularised gradient boosting that uses second order information: [xgboost.py](xgboost/xgboost.py), [libxgboost.py](xgboost/libxgboost.py), [notes](xgboost/README.md)

## Neural Networks

- Feed-forward neural network, trained with backpropagation: [nn.py](neural-network/nn.py), [libnn.py](neural-network/libnn.py), [notes](neural-network/README.md)
- RNN, a network with a hidden state that carries over from step to step: [rnn.py](rnn/rnn.py), [librnn.py](rnn/librnn.py), [notes](rnn/README.md)
- LSTM, an RNN with gates and a cell state for long sequences: [lstm.py](lstm/lstm.py), [liblstm.py](lstm/liblstm.py), [notes](lstm/README.md)
- GRU, a lighter gated RNN: [gru.py](gru/gru.py), [libgru.py](gru/libgru.py), [notes](gru/README.md)

## Hidden Markov Models

An HMM models a system that moves between hidden states and only shows observations. Each of the four problems below has a plain version (small example), a dataset version (parameters estimated from data, with a `dataset` prefix) and a library version.

- Forward-backward, the probability of an observation sequence: [hmmfb.py](hmm-forward-backward/hmmfb.py), [datasethmmfb.py](hmm-forward-backward/datasethmmfb.py), [libhmmfb.py](hmm-forward-backward/libhmmfb.py), [notes](hmm-forward-backward/README.md)
- Viterbi, the most likely hidden state sequence: [hmmviterbi.py](hmm-viterbi/hmmviterbi.py), [datasethmmviterbi.py](hmm-viterbi/datasethmmviterbi.py), [libhmmviterbi.py](hmm-viterbi/libhmmviterbi.py), [notes](hmm-viterbi/README.md)
- Baum-Welch, learning the model parameters from data: [hmmbaumwelch.py](hmm-baum-welch/hmmbaumwelch.py), [datasethmmbaumwelch.py](hmm-baum-welch/datasethmmbaumwelch.py), [libhmmbaumwelch.py](hmm-baum-welch/libhmmbaumwelch.py), [notes](hmm-baum-welch/README.md)
- HMM classifier, one HMM per class: [hmmclassifier.py](hmm-classifier/hmmclassifier.py), [datasethmmclassifier.py](hmm-classifier/datasethmmclassifier.py), [libhmmclassifier.py](hmm-classifier/libhmmclassifier.py), [notes](hmm-classifier/README.md)

## Preprocessing

[preprocessinggeneral.py](preprocessing/preprocessinggeneral.py) is a general template for cleaning, encoding, splitting and scaling a dataset. See [notes](preprocessing/README.md).

## Datasets

- [electricity_cost_dataset.csv](datasets/electricity_cost_dataset.csv), used for linear regression
- [heart_cleveland_upload.csv](datasets/heart_cleveland_upload.csv), used for the SVMs
- [mushrooms.csv](datasets/mushrooms.csv), used for most of the classification algorithms

The `dataset` prefixed HMM scripts read a file named `dataset.csv`, which is not included in the repository.
