import os
from re import M
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def feature_normalization(X):
    """
    Parameters:
    X: array
    =====================
    Returns:
    X as normalized.
    =====================
    Takes the elements of X, calculates their arithmetic mean, standard deviation
    and returns the X with normalized elements.

    """
    X = X.copy()
    X = (X - X.mean()) / X.std()
    return X


def cost_function(X, theta, y):
    """
    Parameters:
    X: array
    theta: array
    y: array
    ====================
    Returns:
    J: Scalar
    Error: array
    ====================
    Takes the features matrix X, parameter matrix theta, real life values y:
    Then returns the error and the result of the cost function.

    """
    m = y.size
    error = np.dot(X, theta) - y
    J = (1 / (2 * m)) * np.dot(error.T, error)

    # X.T.dot(X) means X square in matrices.
    return J, error


def gradient_descent(X, y, theta, alpha, iterations):
    """
    Parameters:
    X: array
    y: array
    theta: array
    alpha: float
    iterations: int
    ====================
    Returns:
    cost_history: List
    theta: array
    ====================
    Applies the gradient descent algorithm for the given arguments.

    """
    cost_history = np.zeros(iterations)
    m = y.size
    for i in range(iterations):
        cost, error = cost_function(X, theta, y)
        theta = theta - (alpha * (1 / m) * np.dot(X.T, error))
        cost_history[i] = cost
    return cost_history, theta


def importdata(x):
    """
    Parameters,
    x: Str

    ====================

    Returns,
    X: array
    y: array

    ====================

    Requirements:
    The script and the csv file must be in the same directory.

    ====================

    Takes the csv file name as an argument and imports the data to python.

    """

    x = str(x)
    x = x + ".csv"
    data = pd.read_csv(x)
    X = data.iloc[:, 0:2]
    y = data.iloc[:, 2]
    return X, y


def stack_ones(x):
    X = np.append(np.ones([x.shape[0], 1], dtype=int), x, axis=1)
    return X


# Plotting the chart
def plotChart(iterations, cost_history):
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), cost_history, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs Iterations')
    plt.style.use('fivethirtyeight')
    plt.show()


def run():

    # Importing the data.
    X, y = importdata("house_practice")

    # Normalizing the data
    X = feature_normalization(X)

    # Appending to X matrices as a column full of ones as parameter theta0
    X = stack_ones(X)

    # Defining the starting point
    theta = np.zeros([X.shape[1]])

    # Computing cost for the starting point
    initial_cost, _ = (cost_function(X, theta, y))
    print(
        f"Our total cost error J with initial theta values of {theta} is: {initial_cost}")

    # Setting gradient descent parameters:
    alpha = 0.01
    iterations = 1000

    # Running Gradient Descent
    cost_history, theta = gradient_descent(X, y, theta, alpha, iterations)

    # Calculating cost for the latest theta values
    final_error, _ = cost_function(X, theta, y)
    print(
        f"After running gradient descent for {iterations} iterations, our final theta values are: {theta}" +
        f" total error for our final theta values are: {final_error}")


run()
