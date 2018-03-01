#Reference
#http://www.cse.iitm.ac.in/~vplab/courses/DVP/PDF/gmm.pdf


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import scipy

# reading data from csv
def read_data():
    df = pd.read_csv("../data/clusters.csv", header=None)
    return df

def normal_distribution(covariance, X, M):
    #print(covariance.shape)
    cova_inverse = np.linalg.inv(covariance)
    D = X - M
    cova_inverse_half = np.linalg.det(covariance) ** (-.5)
    denominator = (2*np.pi)**X.shape[1] / 2
    P = cova_inverse_half * np.exp(-.5 * np.einsum('ij, ij -> i', D, np.dot(cova_inverse, D.T).T)) / denominator
    #print(P.shape)
    return P


def initialisation(df, c):
    n, d = df.shape
    X = df.as_matrix()
    M = X[np.random.choice(n, c, False), :]
    R = np.random.rand(n, c)
    W = [1. / c] * c
    covariance = [np.eye(d)] * c
    return X, M, R, W, covariance


def main():
    df = read_data()
    n, d = df.shape
    c = 3

    X, M, R, W, covariance = initialisation(df, c)

    temp_lh = 0
    for i in range(1000):

        for i in range(c):
            R[:, i] = W[i] * normal_distribution(covariance[i], X, M[i])

        R = (R.T / np.sum(R, axis=1)).T
        R_sum = np.sum(R, axis = 0)

        for i in range(c):
            M[i] = 1.0 / R_sum[i] * np.sum(R[:, i] * X.T, axis=1).T

        for i in range(c):
            D = np.matrix(X - M[i])
            covariance[i] = np.dot(np.multiply(D.T, R[:, i]), D) / R_sum[i]
            W[i] = 1. / n * R_sum[i]

        lh = np.sum(np.log(np.sum(R, axis=1)))
        if temp_lh == lh:
            break
        else:
            temp_lh = lh

    print("Means")
    print(M)
    print("amplitude")
    print(W)
    print("covariance")
    print(covariance)

if __name__ == '__main__':
    main()
