from sklearn.mixture import GaussianMixture
import pandas as pd
from matplotlib import pyplot as plt


def model(df):
    model = GaussianMixture(n_components=3)
    model.fit(df)
    labels = model.predict(df)
    plt.scatter(df[0],df[1], c=labels)
    print("means")
    print(model.means_)
    print("weights")
    print(model.weights_)
    print("covariance")
    print(model.covariances_)


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
    model(df)



if __name__ == "__main__":
    main()