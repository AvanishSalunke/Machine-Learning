# HMM Classifier

Files in this folder:

- [hmmclassifier.py](hmmclassifier.py): an HMM classifier on the mushroom data, step by step with numpy.
- [datasethmmclassifier.py](datasethmmclassifier.py): a more general version that reads `dataset.csv` and handles any number of classes.
- [libhmmclassifier.py](libhmmclassifier.py): library version with `hmmlearn`.

## In simple words

Imagine you know two music styles, jazz and rock. You listen to many jazz pieces and build a "jazz machine" that captures how jazz sequences of notes tend to look. You do the same with rock. When a new piece arrives, you ask both machines: "How likely is it that you would make this piece?" The machine that finds the piece more likely gives you the answer.

That is the HMM classifier. For each class we train one [hidden Markov model](../hmm-forward-backward/README.md) on the sequences of that class. To classify a new sequence, we compute its probability under each model and pick the class with the highest (prior-weighted) probability.

In this repository a mushroom is a **sequence of attributes**: cap shape, then cap surface, then cap colour, and so on, for 22 steps. Each attribute value is a symbol.

## Key words

- **Class-conditional model** $\lambda_c$: the HMM trained only on sequences of class $c$.
- **Prior** $P(c)$: how common the class is in the training data.
- **Log probability**: we add logarithms instead of multiplying tiny probabilities.
- **Multiple sequences**: the model is trained on many sequences at once, not a single long one.

## The maths, step by step

**1. The decision rule** is Bayes' rule with an HMM for each class:

$$\hat{c} = \arg\max_{c}\; P(c)\; P(O \mid \lambda_c)$$

$P(O \mid \lambda_c)$ is the probability that the model of class $c$ produces the observation sequence $O$. It is found with the **forward algorithm** (see [forward-backward](../hmm-forward-backward/README.md)).

**2. Use logarithms** to avoid numbers too small for the computer:

$$\hat{c} = \arg\max_{c}\; \Big[ \ln P(c) + \ln P(O \mid \lambda_c) \Big]$$

**3. The prior** from the training labels, where $N_c$ rows have class $c$:

$$P(c) = \frac{N_c}{N}$$

**4. Training one model** uses [Baum-Welch](../hmm-baum-welch/README.md). With many training sequences $O^{(1)}, \dots, O^{(n)}$ of a class, the E-step is done for every sequence and the expected counts are **added up** before the M-step:

$$\hat{A}_{ij} = \frac{\sum_{s=1}^{n} \sum_{t=1}^{T-1} \xi^{(s)}_t(i, j)}{\sum_{s=1}^{n} \sum_{t=1}^{T-1} \gamma^{(s)}_t(i)}, \qquad \hat{\pi}_i = \frac{1}{n} \sum_{s=1}^{n} \gamma^{(s)}_1(i)$$

$$\hat{B}_j(k) = \frac{\sum_{s} \sum_{t:\, o^{(s)}_t = k} \gamma^{(s)}_t(j)}{\sum_{s} \sum_{t} \gamma^{(s)}_t(j)}$$

**5. Safety for zero probabilities.** If a symbol never appears under a model, its emission probability could drop to exactly 0, and any test sequence containing it would get probability 0. So after every update a tiny number is added to $B$ and the rows are normalised again.

## A small worked example

Suppose that for one test mushroom the two models give:

- edible model: $\ln P(O \mid \lambda_e) = -30.2$, and the prior $P(e) = 0.52$, so $\ln P(e) = -0.65$. Score: $-30.85$.
- poisonous model: $\ln P(O \mid \lambda_p) = -35.8$, and the prior $P(p) = 0.48$, so $\ln P(p) = -0.73$. Score: $-36.53$.

The edible score is larger (closer to zero), so the mushroom is classified as **edible**. The difference of about 5.7 in log scale means the edible model finds this sequence about 300 times more likely.

## How the code works

**[hmmclassifier.py](hmmclassifier.py)**

1. **Load** `../datasets/mushrooms.csv`, replace `?` with the most common value, and convert every attribute value to an integer code, which is the symbol. Each mushroom is one sequence of 22 symbols. Edible = 0, poisonous = 1. **Split** 80/20 with `random_state=42`.
2. **For each class** (seed 42): take the training sequences of that class and create an HMM with 3 hidden states and random $\pi$, $A$, $B$.
3. **Train for 20 Baum-Welch epochs.** The forward and backward passes are done for all sequences at once with array operations. After each update a tiny number is added to $B$ and rows are renormalised. The average negative log likelihood of each epoch is stored for both classes and plotted.
4. **Classify.** For each test sequence, run the forward algorithm with each class model, take the logarithm, and add the log prior. The larger score wins.
5. **Report** accuracy, precision, recall, F1 and the confusion matrix.

**[datasethmmclassifier.py](datasethmmclassifier.py)** is the general version. It reads `dataset.csv`, uses the column named `class` as the target, turns every column into integer symbols, and trains one HMM for each class it finds. With more than two classes it reports macro-averaged precision, recall and F1. The file `dataset.csv` is not included in this repository, so add your own to run it.

## Library version

[libhmmclassifier.py](libhmmclassifier.py) trains one `hmmlearn.hmm.CategoricalHMM` per class (3 states, 20 rounds, using the `lengths` argument so that the library knows where each sequence ends) and scores each test sequence with `model.score`, adding the log prior.

## Strengths and weaknesses

- A natural fit when the **order** of the values matters, like speech, gestures or DNA.
- For the mushroom data the order of attributes is arbitrary, so this is mainly a demonstration of the method. A model that ignores order, like [naive Bayes](../naive-bayes/README.md) or a [tree](../id3/README.md), may work as well or better.
- You must choose the number of hidden states, and training depends on the random start.

## Things to try

- Change the number of hidden states from 3 to 2 or 5.
- Compare accuracy with the [naive Bayes](../naive-bayes/README.md) and [ID3](../id3/README.md) folders.
