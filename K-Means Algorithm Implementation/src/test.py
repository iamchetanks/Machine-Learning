import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy

# reading data from csv
def read_data():
    df = pd.read_csv("../data/clusters.csv", header=None)
    return df

def plot(df):
    fig = plt.figure(figsize=(5,5))
    plt.scatter(df[0],df[1])
    plt.show()

# selecting random points as centroids initially
def select_centroids(df, k):
    c = df.sample(n=k, random_state=200)
    # print(c)
    # print(c.iloc[0][0])
    # plt.scatter(c[0],c[1], c=['r','g','b'])
    # #plt.show()
    # centroids = {}
    # for i in range(k):
    #     centroids[i+1] = [c.iloc[i][0],c.iloc[i][1]]
    # print(centroids[1][1])

    return c


def find_distance(df, centroids, k):
    for i in range(k):
        df['dist_from_centroid_{}'.format(i+1)] = np.sqrt((df[0] - centroids[0].iloc[0]) ** 2 + (df[1] - centroids[1].iloc[i]) ** 2)
    df['close_to'] = df.loc[:,['dist_from_centroid_1', 'dist_from_centroid_2','dist_from_centroid_3']].idxmin(axis = 1)
    df['close_to'] = df['close_to'].str.lstrip('dist_from_centroid_')
    return df

def new_centroid(df, k, centroids):
    #print(centroids)
    for i in range(k):
        centroids.iloc[i][0] = np.mean(df[df['close_to'] == str(i+1)][0])
        centroids.iloc[i][1] = np.mean(df[df['close_to'] == str(i+1)][1])
    #print(centroids)
    return centroids


def main():
    df = read_data()
    plot(df)
    centroids = select_centroids(df, 3)
    #print(centroids)
    df = find_distance(df, centroids, 3)
    #print(df.head())
    while(True):

        new_close_to = copy.deepcopy(df['close_to'])
        #print(df.head())
        centroids = new_centroid(df, 3, centroids)
        df = find_distance(df, centroids, 3)
        if new_close_to.equals(df['close_to']):
            break
    print(centroids)
    plot(centroids)
#if __name__ == '__main__':
  #  main()
main()