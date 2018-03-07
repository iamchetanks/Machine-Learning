import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy

# reading data from csv
def read_data():
    df = pd.read_csv("../data/classification.csv", header=None)
    X_org = np.array(df.iloc[:, 0:3])
    #print(X_org.shape)
    X = X_org.reshape(X_org.shape[0],-1).T
    #print(X.shape)
    Y_org = np.array(df.iloc[:, 3]).T
    Y = Y_org.reshape(Y_org.shape[0], -1).T
    #print(X_org.shape)
    return X, Y

#initializing weights
def initialize_with_zeros(dim):
    wt = np.zeros((dim,1))
    b = 0
    return wt, b


def sigmoid(z):
    s = (1 / (1 + np.exp(-z)))
    return s


def propagate(X,W,b,Y):
    S = np.dot(W.T,X) + b
    #A = sigmoid(S)
    #print(S.shape)
    S[S >= 0.0] = +1
    S[S <= 0.0] = -1
    #print(S)
    # S_list = []
    # for i in S:
    #     for j in i:
    #         if j < 0.0:
    #           S_list.append(-1)
    #         else:
    #             S_list.append(1)
    # S_array = np.array(S_list)
    # S = S_array.reshape(S_array[0],-1)
    N = X.shape[1]
    error  =  abs(Y - S)
    dw = np.dot(X, error.T) / N
    #db = np.sum(error) / N
    cost = np.sum(error) / N
    return dw, error, cost


def main():
    X,Y = read_data()
    learning_rate = 0.01
    no_of_iterations = 1000
    #print(X.shape[0])
    W, b = initialize_with_zeros(X.shape[0])
    #print(W.shape)
    for i in range(no_of_iterations):
        dw, error, cost = propagate(X,W,b,Y)
        W = W - learning_rate * dw
        b = b + learning_rate * error
        print(cost)

if __name__ == '__main__':
    main()
