"""Our own CSP (Common Spatial Patterns), written as a scikit-learn transformer.

Idea: find spatial filters W (weighted sums of electrodes) such that, after
projection, the signal has BIG variance for class A and SMALL variance for
class B (and the opposite). Variance of a band-passed EEG = its power, so these
features directly say "which brain area is active".
"""
import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin


class CSP(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=4):
        # sklearn rule: __init__ only stores parameters, no computation.
        self.n_components = n_components

    def fit(self, X, y):
        """X: (n_epochs, n_channels, n_times), y: labels. Learns self.filters_."""
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("CSP works with exactly 2 classes")

        # 1. Average covariance matrix of each class (channels x channels).
        covs = [self._mean_covariance(X[y == c]) for c in self.classes_]

        # 2. Generalized eigenvalue problem:  C_a w = lambda (C_a + C_b) w
        #    lambda close to 1 -> variance mostly from class a
        #    lambda close to 0 -> variance mostly from class b
        eigenvalues, eigenvectors = eigh(covs[0], covs[0] + covs[1])

        # 3. The most useful filters are at BOTH ends of the sorted eigenvalues.
        #    Sort by distance from 0.5, so the best filters come first.
        order = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
        self.filters_ = eigenvectors[:, order[:self.n_components]]  # W: (channels, n_components)
        return self  # sklearn rule: fit returns self

    def transform(self, X):
        """Project each epoch with W^T and return log-variance features (n_epochs, n_components)."""
        X_csp = np.einsum("ck,ect->ekt", self.filters_, X)  # W^T @ X for every epoch e
        variance = X_csp.var(axis=2)
        # normalize (relative power) then log: makes features closer to Gaussian, which LDA likes
        return np.log(variance / variance.sum(axis=1, keepdims=True))

    @staticmethod
    def _mean_covariance(X):
        covs = []
        for epoch in X:
            cov = np.cov(epoch)                 # (channels, channels)
            covs.append(cov / np.trace(cov))    # trace-normalize: loud epochs don't dominate
        return np.mean(covs, axis=0)


if __name__ == "__main__":
    # Self-check on fake data: class 1 is "loud" on channel 0, class 2 on channel 1.
    rng = np.random.default_rng(0)
    n_epochs, n_channels, n_times = 40, 8, 200
    X = rng.normal(size=(n_epochs, n_channels, n_times))
    y = np.array([1, 2] * (n_epochs // 2))
    X[y == 1, 0] *= 5
    X[y == 2, 1] *= 5

    features = CSP(n_components=2).fit_transform(X, y)
    assert features.shape == (n_epochs, 2)
    # The first feature alone must separate the two classes perfectly.
    f1, f2 = features[y == 1, 0], features[y == 2, 0]
    assert f1.max() < f2.min() or f2.max() < f1.min(), "CSP failed to separate the classes"
    print("CSP self-check OK")
