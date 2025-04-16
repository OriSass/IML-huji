from typing import NoReturn
from linear_regression import LinearRegression
import numpy as np


class PolynomialFitting(LinearRegression):
    """
    Polynomial Fitting using Least Squares estimation
    """
    def __init__(self, k: int):
        """
        Instantiate a polynomial fitting estimator

        Parameters
        ----------
        k : int
            Degree of polynomial to fit
        """
        super().__init__() # todo check if need to check false
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to polynomial transformed samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        X_poly = self.__transform(X)
        super().fit(X_poly, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        X_poly = self.__transform(X)
        return super().predict(X_poly)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        # X_poly = self.__transform(X)
        # print("X_poly shape:", X_poly.shape)
        # print("y shape:", y.shape)
        # assert X_poly.shape[0] == y.shape[0], "Mismatch in prediction input and labels"
        #
        # return super().loss(X_poly, y)
        # x_poly = self.__transform(X)
        # y_pred = super().predict(x_poly)
        # return np.mean((y - y_pred) ** 2)
        return super().loss(X,y)

    def __transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform given input according to the univariate polynomial transformation

        Parameters
        ----------
        X: ndarray of shape (n_samples,)

        Returns
        -------
        transformed: ndarray of shape (n_samples, k+1)
            Vandermonde matrix of given samples up to degree k
        """
        # Ensure X is a 1D array
        # X = np.ravel(X)
        return np.vander(X, N=self.k + 1, increasing=True)
