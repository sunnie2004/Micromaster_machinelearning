import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    return (np.inner(X,Y) + c) ** p

    # raise NotImplementedError



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    row_number_X = X.shape[0]
    row_number_Y = Y.shape[0]
    Kernel_matrix = np.zeros((row_number_X,row_number_Y))

    for i in range(row_number_X):
        for j in range(row_number_Y):
             Kernel_matrix[i][j] = np.exp(-gamma * (np.linalg.norm(X[i,:]) **2 + np.linalg.norm(Y[j,:]) **2 - 2 * np.dot(Y[j,:],X[i,:])))
    return Kernel_matrix
    # raise NotImplementedError
