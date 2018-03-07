import pandas as pd
import numpy as np
import random

# reading data from csv
def read_data():
    df = pd.read_csv("../data/classification.csv", header=None)
    X_org = np.array(df.iloc[:, 0:3])
    #print(X_org.shape)
    intercept = np.ones((X_org.shape[0], 1))
    print(intercept.shape)
    features = np.hstack((intercept, X_org))
    print(features.shape)
    X = features.reshape(features.shape[0],-1).T
    print(X.shape)
    Y_org = np.array(df.iloc[:, 3]).T
    Y = Y_org.reshape(Y_org.shape[0], -1).T
    #print(X_org.shape)


    return X, Y

#initializing weights
def initialize_with_zeros(dim):
    wt = [np.random.uniform(-1,1)] * 4
    wt = np.array([wt])
    #print(wt)
    b = 0
    return wt.T, b


def sigmoid(z):
    s = (1 / (1 + np.exp(-z)))
    return s


def propagate(X,W,Y,learning_rate):
    count1 = 0
    while (True):
        count = 0
        prediction = np.dot(W.T, X)
        X = X.T
        W = W.T

        for i in prediction:
            k = len(i)
            for j in range(k):
                j = random.randrange(0, k, 1)
                # print(Y[0])
                if i[j] > 0:
                    if Y[0][j] == 1:
                        continue
                    else:
                        count += 1
                        W = W - learning_rate * X[j]

                else:
                    if Y[0][j] == 1:
                        count += 1
                        W = W + learning_rate * X[j]
                    else:
                        continue

        X = X.T
        W = W.T
        count1 += 1
        if count == 0:
            print(count1)
            break

    return W


def main():
    X,Y = read_data()
    #print(X.shape)
    #print(Y[0][0])
    W, b = initialize_with_zeros(X.shape[0])
    learning_rate = 0.001
    #print(W.shape)
    W = propagate(X,W,Y,learning_rate)
    print(W)


if __name__ == '__main__':
    main()
