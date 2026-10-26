"""Step 3: evaluate MNE's CSP + LDA before using our custom CSP.

Run: python reference_pipeline.py
Participant 1, imagined left/right fist: develop on runs 4 and 8.
Run 12 is reserved for final testing and is not loaded by this script.
"""
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from preprocess import get_epochs


def main():
    X, y = get_epochs(1, [4, 8])
    labels, counts = np.unique(y, return_counts=True)
    print("Development epochs:", X.shape)
    print("Labels and counts:", dict(zip(labels.tolist(), counts.tolist())))
    print("Run 12 is reserved; it is not used here.")

    # Each validation split fits BOTH steps on its training examples only.
    pipeline = Pipeline([
        ("csp", CSP(n_components=4, log=True)),
        ("classifier", LinearDiscriminantAnalysis()),
    ])
    cross_validation = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42) # reserve 20% for validation in each repetition. Stratified: keep approximately the same label proportions in both groups.
    scores = cross_val_score(pipeline, X, y, cv=cross_validation, scoring="accuracy", error_score="raise") 
    # returns a NumPy array containing 10 accuracy scores: Each score evaluates the whole pipeline, using all four CSP features to predict labels.

    # Compare with predicting the more common label in each training split.
    baseline_scores = []
    for train_indices, validation_indices in cross_validation.split(X, y):
        train_labels, train_counts = np.unique(y[train_indices], return_counts=True)
        majority_label = train_labels[train_counts.argmax()]
        baseline_scores.append(np.mean(y[validation_indices] == majority_label))

    print("Validation accuracies:", np.round(scores, 4))
    print(f"Mean validation accuracy: {scores.mean():.1%}")
    print(f"Mean majority-class baseline: {np.mean(baseline_scores):.1%}")
    print("These are repeated validation splits, not a final test result.")


if __name__ == "__main__":
    main()
