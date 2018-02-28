import numpy as np

from collections import namedtuple

class GMM:

    def __init__(self, k=3, eps=0.000001):
        self.k = k

        self.eps = eps

    '''
        arguments:
            X         = numpy array of the sample points
            max_iters = max number of iterations to test the convergence
    '''

    def em(self, X, maxiterations=500):
        n, d = X.shape

        mu = X[np.random.choice(n, self.k, False), :]

        sig= [np.eye(d)] * self.k

        wt = [1. / self.k] * self.k

        R = np.zeros((n, self.k))

        likelihoods = []

        P = lambda mu, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-X.shape[1] / 2.) \
                          * np.exp(-.5 * np.einsum('ij, ij -> i', \
                                                   X - mu, np.dot(np.linalg.inv(s), (X - mu).T).T))

        while len(likelihoods) < maxiterations:

            for k in range(self.k):

                R[:, k] = wt[k] * P(mu[k], sig[k])

            likelihood = np.sum(np.log(np.sum(R, axis=1)))

            likelihoods.append(likelihood)

            R = (R.T / np.sum(R, axis=1)).T

            N_ks = np.sum(R, axis=0)

            for k in range(self.k):

                mu[k] = 1. / N_ks[k] * np.sum(R[:, k] * X.T, axis=1).T

                x_mu = np.matrix(X - mu[k])

                sig[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T, R[:, k]), x_mu))

                wt[k] = 1. / n * N_ks[k]

            if len(likelihoods) < 2: continue

            if np.abs(likelihood - likelihoods[-2]) < self.eps: break

        self.params = namedtuple('params', ['mu', 'sig', 'wt', 'likelihoods', 'numiterations'])

        self.params.mu = mu

        self.params.sig = sig

        self.params.wt = wt

        self.params.likelihoods = likelihoods

        self.params.numiterations = len(likelihoods)

        return self.params

    def amp(self, x):
        p = lambda mu, s : np.linalg.det(s) ** - 0.5 * (2 * np.pi) **\
                (-len(x)/2) * np.exp( -0.5 * np.dot(x - mu , \
                        np.dot(np.linalg.inv(s) , x - mu)))
        calvalue = np.array([w * p(mu, s) for mu, s, w in \
            zip(self.params.mu, self.params.sig, self.params.wt)])
        return calvalue/np.sum(calvalue)

if __name__ == "__main__":

    X = np.genfromtxt("../data/clusters.csv", delimiter=',')

    gmm = GMM(3, 0.000001)

    params = gmm.em(X, maxiterations=500)

    print("Mean=", params.mu)

    print("Amp=", gmm.amp(np.array([1, 2])))

    print("Covariance=", params.sig)
