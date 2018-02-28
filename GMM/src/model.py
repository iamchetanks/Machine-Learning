import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import scipy

# reading data from csv
def read_data():
    df = pd.read_csv("../data/clusters.csv", header=None)
    return df

# def amp(R1, R2, R3):
#     N = (np.sum(R1) + np.sum(R2) + np.sum(R3))
#     amp1 = np.sum(R1) / N
#     amp2 = np.sum(R2) / N
#     amp3 = np.sum(R3) / N
#     return amp1, amp2, amp3

# def mean(R1, R2, R3, X):
#     M1 = np.dot(R1, X) / np.sum(R1)
#     #print(M1.shape)
#     M2 = np.dot(R2, X) / np.sum(R2)
#     M3 = np.dot(R3, X) / np.sum(R3)
#     return M1, M2, M3

def normal_distribution(covariance, X, M):
    #print(covariance.shape)
    cova_inverse = np.linalg.inv(covariance)
    D = X - M
    cova_inverse_half = np.linalg.det(covariance) ** (-.5)
    denominator = (2*np.pi)**X.shape[1] / 2
    P = cova_inverse_half * np.exp(-.5 * np.einsum('ij, ij -> i', D, np.dot(cova_inverse, D.T).T)) / denominator
    #print(P.shape)
    return P

def main():
    df = read_data()
    n, d = df.shape
    #print(df.head())
    X = df.as_matrix()
    #X = X.transpose()
    M = X[np.random.choice(n,3,False), :]
    R = np.random.rand(n, 3)
    # N = (np.sum(R[0]) + np.sum(R[1]) + np.sum(R[2]))
    # for i in range(3):
    #     amp[i] = amp(R[:,i], N)

    W = [1. / 3] * 3
    covariance = [np.eye(d)] * 3
    for i in range(100):
        for i in range(3):
            R[:, i] = W[i] * normal_distribution(covariance[i], X, M[i])

        R = (R.T / np.sum(R, axis=1)).T
        #print(R)
        R_sum = np.sum(R, axis = 0)
        #print(R_sum[0])
        for i in range(3):
            M[i] = 1.0 / R_sum[i] * np.sum(R[:, i] * X.T, axis=1).T


        for i in range(3):
            D = np.matrix(X - M[i])
            covariance[i] = np.dot(np.multiply(D.T, R[:, i]), D) / R_sum[i]
            W[i] = 1. / n * R_sum[i]

    print("Means")
    print(M)
    print("amp")
    print(W)
    print("covariance")
    print(covariance)

if __name__ == '__main__':
    main()
