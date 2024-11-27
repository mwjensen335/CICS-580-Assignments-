import numpy as np
import numpy.linalg as la
from sklearn.linear_model import LinearRegression
import unittest

############################################################
# Problem 1: Gauss-Jordan Elimination
############################################################

import numpy as np

def gauss_jordan(A):
    """
    Inverts a given square matrix A using Gauss-Jordan elimination.

    Parameters:
    A (np.ndarray): A square, two-dimensional NumPy array.

    Returns:
    np.ndarray or None: The inverse of matrix A if it is invertible, else None.
    """
    n = A.shape[0]
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square matrix.")

    # Create an augmented matrix [A | I]
    augmented_matrix = np.hstack((A.astype(float), np.eye(n)))

    # Perform Gauss-Jordan elimination
    for i in range(n):
        # Pivoting: Find the row with the maximum element in the current column
        max_row = np.argmax(abs(augmented_matrix[i:, i])) + i
        if augmented_matrix[max_row, i] == 0:
            # If the pivot is zero, the matrix is singular (non-invertible)
            return None
        
        # Swap the current row with the max row for numerical stability
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]

        # Normalize the pivot row
        pivot = augmented_matrix[i, i]
        augmented_matrix[i] = augmented_matrix[i] / pivot

        # Eliminate the other entries in the current column
        for j in range(n):
            if j != i:
                factor = augmented_matrix[j, i]
                augmented_matrix[j] -= factor * augmented_matrix[i]

    # The inverse of A is now in the augmented part of the matrix
    A_inv = augmented_matrix[:, n:]
    return A_inv


    
############################################################
# Problem 2: Ordinary Least Squares Linear Regression
############################################################

def linear_regression_inverse(X,y):
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ X.T @ y
    return beta
    
def linear_regression_moore_penrose(X,y):
    X_pseudo_inv= np.linalg.pinv(X)
    beta = X_pseudo_inv @ y
    return beta
    
def generate_data(n,m):
    """
        Generates a synthetic data matrix X of size n by m
        and a length n response vector.

        Input:
            n - Integer number of data cases.
            m - Integer number of independent variables.

        Output:
            X - n by m numpy array containing independent variable
                observasions.
            y - length n numpy array containg dependent variable
                observations.
    """
    X = np.random.randn(n, m)
    beta = np.random.randn(m)
    epsilon = np.random.randn(n)*0.5
    y = np.dot(X, beta) + epsilon

    return X, y


if __name__=="__main__":
    # test gauss-jordan elimination
    X = np.random.randn(3, 3)
    print(np.allclose(gauss_jordan(X), la.inv(X)))

    # test linear regression
    X, y = generate_data(10, 3)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, y)
    beta = lr.coef_
    print(np.allclose(linear_regression_inverse(X, y), beta))
    print(np.allclose(linear_regression_moore_penrose(X, y), beta))
