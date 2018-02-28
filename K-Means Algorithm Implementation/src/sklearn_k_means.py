from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt
from numpy import genfromtxt


def model(df):
    model = KMeans(n_clusters=3)
    model.fit(df)
    label = model.predict(df)
    print(label)
    centroids = model.cluster_centers_
    print(centroids)
    return centroids


def plot(df, centroids):
    ax = df.plot.scatter(x=0, y=1, color='grey', label='Data Points')
    centroids.plot.scatter(x=0, y=1, color='blue', s=50, label='Centroids', ax=ax)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Sklearn Graph")
    plt.show()


def read_data():
    df = pd.read_csv("../data/clusters.csv", header=None)
    return df


def main():
    df = read_data()
    centroids = model(df)
    centroids = pd.DataFrame(data=centroids[:,:])
    plot(df, centroids)


if __name__ == "__main__":
    main()