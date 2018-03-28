import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
from numpy.linalg import inv

# reading data from csv
def read_data():
    df = pd.read_csv("../data/linear-regression.txt", header=None)
    X_org = np.array(df.iloc[:, 0:2])
    #print(X_org.shape)
    X = X_org.reshape(X_org.shape[0],-1).T
    print(X.shape)
    Y_org = np.array(df.iloc[:, 2]).T
    Y = Y_org.reshape(Y_org.shape[0], -1).T
    print(Y.shape)
    return X, Y


def fit(X, Y):
    W = inv(np.dot(X, X.T))
    k = X * Y
    print(k.shape)
    print(W)
    return W

def main():
    X,Y = read_data()
    W = fit(X, Y)
if __name__ == '__main__':
    main()
