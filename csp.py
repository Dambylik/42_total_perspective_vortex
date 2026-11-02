"""Common Spatial Patterns algotithmes (CSP), written as a scikit-learn transformer.

"""
import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin


class CSP(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=4):
        self.n_components = n_components

    def fit(self, X, y):
        """X: (n_epochs, n_channels, n_times), y: labels. Learns self.filters_."""
        self.classes_ = np.unique(y) # self.classes_ = [1, 2]
        if len(self.classes_) != 2:
            raise ValueError("CSP works with exactly 2 classes")

        # 1. Average covariance matrix of each class (channels x channels)
        covs = []
        for label in self.classes_:
            class_epochs = X[y == label]
            average_covariance = self._mean_covariance(class_epochs)
            covs.append(average_covariance)

        # 2. Generalized eigenvalue calculus
        eigenvalues, eigenvectors = eigh(covs[0], covs[0] + covs[1])

        # 3. The most useful filters are at both ends of the sorted eigenvalues.
        order = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
        self.filters_ = eigenvectors[:, order[:self.n_components]]  # W: (channels, n_components)
        return self

    def transform(self, X):
        """Project each epoch with W^T and return log-variance features (n_epochs, n_components)."""
        X_csp = np.einsum("ck,ect->ekt", self.filters_, X) # c = channel, k = filter, e = epoch, t = time
        variance = X_csp.var(axis=2)
        return np.log(variance / variance.sum(axis=1, keepdims=True))

    @staticmethod
    def _mean_covariance(X):
        covs = []
        for epoch in X:
            cov = np.cov(epoch)
            covs.append(cov / np.trace(cov))
        return np.mean(covs, axis=0)

