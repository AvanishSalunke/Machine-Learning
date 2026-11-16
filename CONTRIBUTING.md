# Contributing

Thanks for wanting to help. This project is a collection of machine learning algorithms, each with plain code, a library version where one makes sense, and a README that explains it. Fixes, new algorithms and better explanations are all welcome.

## Ways to help

- Fix a bug or a wrong formula in the code or in a README.
- Make an explanation simpler or clearer.
- Add one of the algorithms from the [roadmap](ROADMAP.md).
- Add a missing library version for an algorithm that has only one.

If you plan something big, open an issue first so that nobody works on the same thing twice.

## Setting up

1. Fork the repository on GitHub with the Fork button.
2. Clone your fork and go into it:

```bash
git clone https://github.com/<your-username>/Machine-Learning.git
cd Machine-Learning
```

3. Install the requirements (see the [README](README.md) for a virtual environment):

```bash
pip install -r requirements.txt
```

## Making a change

1. Make a new branch for your work. Do not work on `main`.

```bash
git checkout -b add-knn
```

2. Make your changes, and run the scripts you touched from inside their own folder to check that they work.
3. Commit with a short message that says what changed:

```bash
git add knn
git commit -m "add k nearest neighbours"
```

4. Push the branch to your fork:

```bash
git push origin add-knn
```

5. On GitHub, open a pull request from your branch to `main` of this repository. Say in a few lines what you changed and why.

Keep each pull request about one thing. Small pull requests are reviewed faster.

## How an algorithm folder should look

Every algorithm lives in its own folder with a short lowercase name that uses hyphens, for example `k-means` or `linear-regression`. Inside it:

- the main code file, for example `kmeans.py`. It should follow the algorithm step by step with numpy, so that a reader can see how it works.
- the library version, with a `lib` prefix, for example `libkmeans.py`. Add it when a library offers the same algorithm. If no library has it, skip it. If an algorithm needs more than one variant, add more files and explain each one in the README.
- a `README.md` that explains the algorithm.

## How the README of an algorithm should look

Write for a reader who is learning. Use simple words, but be technical and correct. A good README has these parts:

1. The files in the folder.
2. The idea in simple words, with a small everyday comparison if it helps.
3. The maths: every formula, with a line saying what each symbol means.
4. A small worked example with real numbers.
5. How the code works, step by step, and the settings it uses.
6. How the library version differs.
7. Strengths, weaknesses and things to try.

Write formulas in LaTeX between dollar signs, because GitHub shows them nicely. Look at [id3](id3/README.md) or [adaboost](adaboost/README.md) for examples.

## Code style

- Use the datasets in the [datasets](datasets) folder and read them with the path `../datasets/<name>.csv`, so that the script works when run from its own folder.
- Use fixed random seeds (for example `random_state=42`) so results can be repeated.
- Print the results that matter: for classification, accuracy, precision, recall, F1 and the confusion matrix. For regression, MAE, MSE, RMSE and R2.
- Keep the code simple and readable. Clear names are better than clever tricks.
- Do not commit large files, virtual environments or output images.

## Before you open the pull request

- The scripts run without errors from inside their folder.
- The folder has the README, and the formulas in it match the code.
- New algorithms are added to the list in the main README.
- If your algorithm was on the roadmap, tick it off in [ROADMAP.md](ROADMAP.md).

## Reporting problems

Open an issue and describe what you expected, what happened, and the steps to repeat it. Add the error message if there is one.

By contributing, you agree that your work is shared under the [MIT License](LICENSE) of this project.
