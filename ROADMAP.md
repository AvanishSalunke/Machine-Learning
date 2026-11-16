# Roadmap

These are the algorithms that are still to be added. Each one will get its own folder with the code, a library version where a library offers it, and a README with the formulas and a worked example, like the algorithms that are already done. The note after each item shows the planned versions: "code and library" means two files, "code only" means no library has it, "library only" means only the library version makes sense.

Want to take one? Read [CONTRIBUTING.md](CONTRIBUTING.md) and open a pull request.

## A. Maths prerequisites

- [ ] Solve a linear system (matrix inverse and Gaussian elimination) - code and library
- [ ] Eigen-decomposition - code and library
- [ ] SVD and the pseudoinverse - code and library
- [ ] PCA (covariance, eigenvectors, projection) - code and library
- [ ] MLE for Bernoulli and Gaussian parameters - code and library
- [ ] MAP estimation - code only
- [ ] Weight initialisation (He and Xavier variance rules) - code and library

## B. Decision trees

- [x] ID3, gain ratio, Gini tree and misclassification-error tree - done
- [ ] CART measure 2(Ny/N)(Nn/N) Σ|P(c|Sy) − P(c|Sn)| - code only
- [ ] Best split for continuous attributes (sort, scan thresholds) - code and library
- [ ] Reduced-error pruning - code only
- [ ] Candidate elimination and Find-S (version space) - code only
- [ ] MDL tree-cost comparison - code only
- [ ] Missing values and cost-sensitive gain - code only

## C. Probabilistic

- [x] Naive Bayes - done
- [ ] Bayesian belief network inference (heart-disease example) - code and library

## D. Rule-based

- [ ] Ordered rule set (coverage, accuracy, default class) - code only
- [ ] Sequential covering with FOIL gain - code only
- [ ] RIPPER (library `wittgenstein`) - library only
- [ ] Tree to rules (C4.5rules style) - code and library
- [ ] 1R - code only

## E. Ensembles

- [x] Bagging, random forest, AdaBoost, gradient boosting, XGBoost - done
- [ ] Stacking - code and library

## F. Evaluation

- [x] Confusion matrix, accuracy, precision, recall, F1 - done (printed by the classification scripts)
- [ ] ROC curve and AUC - code and library
- [ ] K-fold cross-validation - code and library
- [ ] Comparing classifiers (z-test, McNemar, paired t-test) - code and library
- [ ] Kappa, sensitivity/specificity, R2 - code and library

## G. Regression and linear models

- [x] Linear regression (gradient descent), logistic regression, preprocessing - done
- [ ] Normal equation / ordinary least squares - code and library
- [ ] Polynomial regression - code and library
- [ ] One-vs-all multiclass - code and library
- [ ] Ridge, lasso and elastic net - code and library
- [ ] Subset selection (best subset, forward, backward) - code and library
- [ ] Gradient-descent variants and optimisers - code and library

## H. Neural networks

- [x] Two-layer network with forward and back propagation - done
- [ ] McCulloch-Pitts AND/OR neuron - code only
- [ ] Computation-graph backprop numerical (J = 3(a + bc), and f = (x + y)z) - code and library
- [ ] Batch normalization - code and library

## I. SVM

- [x] Linear SVM, kernel SVM (Gaussian, polynomial) - done
- [ ] SVM one-vs-all - code and library
- [ ] SVM dual form and the worked w, b numerical - code and library

## J. Clustering and other

- [ ] K-means - code and library
- [ ] Hierarchical clustering (Dmin, Dmax, Davg linkage) - code and library
- [ ] kNN - code and library

## K. CNN

- [ ] Convolution forward (stride, padding) - code and library
- [ ] Max pooling - code and library
- [ ] Output-size and parameter-count calculator - code only
- [ ] CNN backprop (conv gradient, rotated filter, pooling routing) - code and library
- [ ] CNN in Keras - library only

## L. Sequence models

- [x] Naive RNN, LSTM, GRU, HMM forward-backward, Viterbi, Baum-Welch, HMM classifier - done
- [ ] Deep and bidirectional RNN - code and library
- [ ] Markov chain (sequence probability, mini-forward, stationary distribution) - code only
- [ ] PageRank - code and library
- [ ] Gibbs sampling - code only
- [ ] HMM parameters by counting (when states are known) - code only

## M. ELM paper and GA

- [ ] Extreme Learning Machine (random hidden layer, β = H†T) - code only
- [ ] Genetic and memetic algorithm (selection, crossover, mutation) - code and library
- [ ] Time-domain feature extraction (mean, SD, RMS and so on) - code only
- [ ] Grid search - code and library
