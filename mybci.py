"""Brain Computer Interface CLI.

    python mybci.py <subject> <run> train     cross-validate + save a model
    python mybci.py <subject> <run> predict   replay <run> as a stream and predict it
    python mybci.py                           evaluate the 6 experiments on all 109 subjects

Train / test split (no data leakage): the model is trained on the OTHER runs of the
same experiment, and tested on <run>, which it has never seen.
"""
import os
import sys
import time

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from csp import CSP
from preprocess import EXPERIMENTS, find_experiment, get_epochs

N_SUBJECTS = 109
MODEL_DIR = "./models"
MAX_DELAY = 2.0  # seconds allowed to predict one chunk (from the subject)


def build_pipeline():
    return Pipeline([
        ("csp", CSP(n_components=4)),               # 64 channels x 321 times -> 4 features
        ("lda", LinearDiscriminantAnalysis()),       # simple linear classifier, standard for CSP
    ])


def split_runs(run):
    """Training runs = the other runs of the experiment containing `run`."""
    runs = find_experiment(run)
    return [r for r in runs if r != run]


def model_path(subject, run):
    return os.path.join(MODEL_DIR, f"subject{subject:03d}_run{run:02d}.joblib")


def train(subject, run):
    X, y = get_epochs(subject, split_runs(run))
    pipeline = build_pipeline()

    # 10 random splits (different each time: no random_state) -> honest estimate.
    cv = ShuffleSplit(n_splits=10, test_size=0.2)
    scores = cross_val_score(pipeline, X, y, cv=cv)
    print(np.round(scores, 4))
    print(f"cross_val_score: {scores.mean():.4f}")

    pipeline.fit(X, y)  # final model = trained on ALL training data
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, model_path(subject, run))


def predict(subject, run):
    path = model_path(subject, run)
    if not os.path.exists(path):
        sys.exit(f"No model found, run first: python mybci.py {subject} {run} train")
    pipeline = joblib.load(path)
    X, y = get_epochs(subject, [run])  # never-learned data

    print("epoch nb: [prediction] [truth] equal?")
    correct = 0
    # ponytail: the "stream" is a loop that feeds one epoch at a time, as if it just arrived.
    for i, (epoch, truth) in enumerate(zip(X, y)):
        start = time.time()
        prediction = pipeline.predict(epoch[np.newaxis])[0]  # model expects a batch -> add axis
        delay = time.time() - start
        if delay > MAX_DELAY:
            print(f"WARNING: epoch {i} took {delay:.2f}s (> {MAX_DELAY}s)")
        correct += prediction == truth
        print(f"epoch {i:02d}:\t[{prediction}]\t[{truth}] {prediction == truth}")
    print(f"Accuracy: {correct / len(y):.4f}")


def evaluate(subject, runs):
    """Train on all runs but the last one, return accuracy on the last one."""
    X_train, y_train = get_epochs(subject, runs[:-1])
    X_test, y_test = get_epochs(subject, runs[-1:])
    return build_pipeline().fit(X_train, y_train).score(X_test, y_test)


def run_all():
    means = []
    for exp, runs in enumerate(EXPERIMENTS):
        scores = []
        for subject in range(1, N_SUBJECTS + 1):
            try:
                score = evaluate(subject, runs)
            except Exception as error:  # a few PhysioNet recordings are broken
                print(f"experiment {exp}: subject {subject:03d}: skipped ({error})")
                continue
            scores.append(score)
            print(f"experiment {exp}: subject {subject:03d}: accuracy = {score:.4f}")
        means.append(np.mean(scores))

    print(f"\nMean accuracy of the six different experiments for all {N_SUBJECTS} subjects:")
    for exp, mean in enumerate(means):
        print(f"experiment {exp}:\t\taccuracy = {mean:.4f}")
    print(f"\nMean accuracy of 6 experiments: {np.mean(means):.4f}")


def main():
    args = sys.argv[1:]
    if not args:
        return run_all()
    if len(args) != 3 or args[2] not in ("train", "predict") \
            or not args[0].isdigit() or not args[1].isdigit():
        sys.exit(__doc__)
    subject, run, mode = int(args[0]), int(args[1]), args[2]
    if not 1 <= subject <= N_SUBJECTS:
        sys.exit(f"subject must be between 1 and {N_SUBJECTS}")
    try:
        find_experiment(run)
    except ValueError as error:
        sys.exit(str(error))
    train(subject, run) if mode == "train" else predict(subject, run)


if __name__ == "__main__":
    main()
