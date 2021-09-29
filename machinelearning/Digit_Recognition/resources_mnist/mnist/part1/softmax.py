import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    #YOUR CODE HERE

    row_number = X.shape[0]
    sum_col = np.zeros(row_number)  #sum of e (ethta8x/t-c)
    C = np.zeros(row_number)      #create the  fixed amount C array
    multi = np.matmul(theta,X.T) / temp_parameter  #get the multiple of theta*X/T

    C = np.array(np.max(multi,axis=0))         #find the max in every column,THEN get C
    for i in range(row_number):
        multi[:,i] = multi[:,i]-C[i]                      # theta * x/T -C

    multi = np.exp(multi)      #exp
    sum_col = multi.sum(axis=0)
    for i in range(row_number):
        multi[:,i] = multi[:,i] /sum_col[i]

    return multi
    # raise NotImplementedError

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    #YOUR CODE HERE
    # sum_error = 0
    # H_X = compute_probabilities(X,theta,temp_parameter)
    # row_number_X = X.shape[0]   #x row
    # row_number_theta = theta.shape[0]
    # for i in range(row_number_X):     #check the condition for label,nesting loop cycle isn't good, so need to find better solution
    #     for j in range(row_number_theta):
    #         if Y[i] == j and H_X[i][j] !=0:
    #             sum_error += np.log(H_X[i][j])
    # J = - sum_error/row_number_X + lambda_factor /2 * np.sum(theta**2)    #
    #
    # return J
    sum_error = 0
    H_X = compute_probabilities(X,theta,temp_parameter)
    row_number_X = X.shape[0]   #x row
    row_number_theta = theta.shape[0]
       #check the condition for label
    for i in range(row_number_X):
        if H_X[Y[i]][i] != 0:
            sum_error += np.log(H_X[Y[i]][i])

    J = - sum_error/row_number_X + lambda_factor /2 * np.sum(theta**2)    #

    return J
    # raise NotImplementedError

def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    #YOUR CODE HERE
    H_X = compute_probabilities(X,theta,temp_parameter)  #get the probability
    row_number_H_X,col_number_H_X = H_X.shape    #row:labelnumber, col:example number
    label_number,feature_number = theta.shape
    theta_derivation = np.empty([label_number,feature_number])  #store theta derivation
    M = sparse.coo_matrix(([1] * col_number_H_X, (Y, range(col_number_H_X))), shape=(row_number_H_X, col_number_H_X)).toarray() #construct coo matrix to store label Y-X[i]
    theta_derivation = -np.matmul(M-H_X,X)/(temp_parameter*col_number_H_X) + lambda_factor*theta  #derivation after gradient descent
    theta -= alpha * theta_derivation  #update theta
    return theta
    # raise NotImplementedError

def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    #YOUR CODE HERE
    test_y_mod3 = np.remainder(test_y,3)
    train_y_mod3 = np.remainder(train_y,3)
    return train_y_mod3,test_y_mod3
    # raise NotImplementedError

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    #YOUR CODE HERE

    estimate_Y = get_classification(X,theta,temp_parameter) #get predict Y ,return 0-9 number
    #estimate_Y_mod3 = update_y(estimate_Y,0)  #change to mode3
    estimate_Y_mod3 = np.remainder(estimate_Y, 3)
    error_number = np.argwhere(estimate_Y_mod3!=Y).size  #compare array
    total_data = len(Y)
    return error_number/total_data

    #raise NotImplementedError

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
