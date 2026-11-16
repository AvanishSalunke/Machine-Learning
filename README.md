# Machine Learning Lab

This repository holds the code for my Machine Learning lab (Semester 5). It collects the common machine learning algorithms in one place. Most algorithms come in two versions:

- a from-scratch implementation, where the algorithm is written by hand
- a library version, where the same algorithm is done with a library such as scikit-learn, with a `lib` prefix on the file name

Below, each algorithm has a short description and links to its files.

## Linear and Logistic Regression

Linear regression fits a straight line (or hyperplane) to continuous data by minimising the squared error between predictions and actual values. Logistic regression takes the same linear idea but passes the output through a sigmoid function, so it predicts the probability of a class and is used for classification.

- [linearregression.py](linearregression.py) and [liblinearregression.py](liblinearregression.py)
- [logisticregression.py](logisticregression.py) and [liblogisticregression.py](liblogisticregression.py)

## Naive Bayes

A probabilistic classifier based on Bayes' theorem. It assumes that features are independent of each other given the class, which is rarely true but works surprisingly well, especially on categorical data.

- [naivebayes.py](naivebayes.py) and [libnaivebayes.py](libnaivebayes.py)

## Decision Trees

A decision tree splits the data again and again on the feature that best separates the classes, until the leaves are mostly pure. The algorithms differ in how they pick the best split.

- ID3 chooses the split with the highest information gain (the biggest drop in entropy): [id3.py](id3.py), [id3_3.py](id3_3.py), [id3tree.py](id3tree.py)
- Gain ratio fixes ID3's bias towards features with many values by dividing information gain by the split information: [gainratio.py](gainratio.py)
- Gini index measures how often a random sample would be mislabelled if labelled by the class distribution in the node. Lower is better: [giniindex.py](giniindex.py)
- Misclassification error uses the fraction of wrongly classified samples in a node as the impurity: [misclassificationerror.py](misclassificationerror.py)

## Support Vector Machines

SVMs find the boundary that separates the classes with the largest possible margin. A linear SVM draws a straight boundary. Kernel SVMs (polynomial, RBF and so on) map the data into a higher dimension so that non-linear boundaries become possible.

- Linear: [svmlinear.py](svmlinear.py) and [libsvmlinear.py](libsvmlinear.py)
- Polynomial kernel: [svmpolynomial.py](svmpolynomial.py)
- Kernel: [svmkernel.py](svmkernel.py) and [libsvmkernel.py](libsvmkernel.py)

## Ensemble Methods

Ensembles combine many weak models into one strong model.

- Bagging trains many models on random resamples of the data and averages them, which reduces variance: [bagging.py](bagging.py) and [libbagging.py](libbagging.py)
- Random forest is bagging with decision trees, plus a random subset of features at each split: [randomforest.py](randomforest.py) and [librandomforest.py](librandomforest.py)
- AdaBoost trains models one after another, giving more weight to samples the previous model got wrong: [adaboost.py](adaboost.py) and [libadaboost.py](libadaboost.py)
- Gradient boosting builds each new model to fit the errors (gradients of the loss) left by the models before it: [gradientboosting.py](gradientboosting.py) and [libgradientboosting.py](libgradientboosting.py)
- XGBoost is an optimised and regularised form of gradient boosting that is fast and widely used: [xgboost.py](xgboost.py) and [libxgboost.py](libxgboost.py)

## Neural Networks

- Feed-forward neural network: layers of neurons trained with backpropagation and gradient descent. [nn.py](nn.py) and [libnn.py](libnn.py)
- RNN: a network with a loop that carries a hidden state from one step to the next, so it can handle sequences. [rnn.py](rnn.py) and [librnn.py](librnn.py)
- LSTM: an RNN with gates that decide what to remember and forget, which helps with long sequences. [lstm.py](lstm.py) and [liblstm.py](liblstm.py)
- GRU: a lighter version of the LSTM with fewer gates. [gru.py](gru.py) and [libgru.py](libgru.py)

## Hidden Markov Models

An HMM models a system that moves between hidden states, where we only see the outputs (observations) the states produce. Four standard problems are covered, each with a plain version and a version that runs on a dataset (`dataset` prefix), plus library versions.

- Forward-backward computes the probability of an observation sequence: [hmmfb.py](hmmfb.py), [datasethmmfb.py](datasethmmfb.py), [libhmmfb.py](libhmmfb.py)
- Viterbi finds the most likely sequence of hidden states: [hmmviterbi.py](hmmviterbi.py), [datasethmmviterbi.py](datasethmmviterbi.py), [libhmmviterbi.py](libhmmviterbi.py)
- Baum-Welch learns the model parameters from data (an expectation-maximisation method): [hmmbaumwelch.py](hmmbaumwelch.py), [datasethmmbaumwelch.py](datasethmmbaumwelch.py), [libhmmbaumwelch.py](libhmmbaumwelch.py)
- HMM classifier trains one HMM per class and picks the class whose model explains a sequence best: [hmmclassifier.py](hmmclassifier.py), [datasethmmclassifier.py](datasethmmclassifier.py), [libhmmclassifier.py](libhmmclassifier.py)

## Other Files

- [preprocessinggeneral.py](preprocessinggeneral.py) has general data preprocessing steps.
- [practice.py](practice.py) is for trying things out.

## Datasets

- [electricity_cost_dataset.csv](electricity_cost_dataset.csv)
- [heart_cleveland_upload.csv](heart_cleveland_upload.csv)
- [mushrooms.csv](mushrooms.csv)
