
import numpy as np
from typing import NoReturn


class LinearRegression:
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem

    Attributes
    ----------
    fitted_ : bool
        Indicates if estimator has been fitted. Set to True in ``self.fit`` function

    include_intercept_: bool
        Should fitted model include an intercept or not

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by linear regression. To be set in
        `LinearRegression.fit` function.
    """

    def __init__(self, include_intercept: bool = True):
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not
        """
        self._fitted: bool = False
        self._include_intercept: bool = include_intercept;
        self._coefs: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        if self._include_intercept:
            X = np.column_stack((np.ones(X.shape[0]), X))  # Add intercept term

            # Solve the normal equations: (X^T X)^{-1} X^T y

        # The heart of linear regression
        # the result is a vector w = (w0,w1,...,wn)
        # it's meaning is the weight's that help us predict.
        self._coefs = np.linalg.pinv(X.T @ X) @ X.T @ y
        self._fitted = True

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
        if not self._fitted:
            raise ValueError("Model must be fitted before predicting.")

        if self._include_intercept:
            X = np.column_stack((np.ones(X.shape[0]), X))

        return X @ self._coefs

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under **mean squared error (MSE) loss function**

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
        y_pred = self.predict(X)

        # print("Predictions (first few values):", y_pred[:5])  # Inspect first few predictions
        # print("Are there any NaN values in predictions?", np.isnan(y_pred).sum())  # Count NaN values

        # Check for NaNs in target (y) and predictions (y_pred)
        if np.isnan(y).any() or np.isnan(y_pred).any():
            print("NaN values found in target or predictions!")
            print("NaN values in target (y):", np.isnan(y).sum())
            print("NaN values in predictions (y_pred):", np.isnan(y_pred).sum())

        # Check for any small values in predictions that could cause issues
        # if np.any(y_pred < 1e-10):  # Adjust threshold as necessary
        #     print("Warning: Small values detected in predictions.")
        #     print(y_pred[np.abs(y_pred) < 1e-10])  # Print values below the threshold

        # Calculate MSE loss
        return np.mean((y - y_pred) ** 2)
