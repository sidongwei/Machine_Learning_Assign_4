import numpy as np
import matplotlib.pyplot as plt

def Gaussian(x, y, sigma):
    yv, xv = np.meshgrid(y ** 2, x ** 2)
    return np.exp(-(xv - 2 * np.dot(x, y.T) + yv) / (2 * sigma))

def Gaussian_deriv(x, y, sigma):
    yv, xv = np.meshgrid(y ** 2, x ** 2)
    mesh = -(xv - 2 * np.dot(x, y.T) + yv) / sigma
    return (mesh + 1) * np.exp(mesh / 2) / sigma

def plot_Gaussian(T, n, func, avg, sigma, seed):            # func: covariance function, avg: mean
    t = len(T)
    m = avg(T)
    K = func(T, T, sigma)

    L = np.linalg.cholesky(K+1e-5*np.eye(t))            # cholesky decomposition
    X = np.zeros((t, n))
    for i in range(n):
        np.random.seed(i*seed+2*i+5*seed+1)         # guarantee same seed for same parameter same line
        x = np.dot(L, np.random.normal(0, 1, t).reshape(t, 1))
        x = m + x
        X[:, i] = x.reshape(t)
    plt.plot(T, X)          # plot a line

def zero_initial(t):            # function to produce 0 mean value
    n = len(t)
    return np.zeros((n, 1))

if __name__ == '__main__':
    T = np.arange(-1, 1, 2/1500).reshape(1500,1)
    sigma = [0.1,1,10]
    seed = [10,29,1995]         # random seeds
    for i in range(3):
        plt.title('sigma=%.1f' % sigma[i])              # plot normal
        plot_Gaussian(T, 10, Gaussian, zero_initial, sigma[i],seed[i])
        plt.savefig(('sigma=%.1f.png' % sigma[i]))
        plt.cla()

        plt.title('sigma=%.1f' % sigma[i])              # plot derivative
        plot_Gaussian(T, 10, Gaussian_deriv, zero_initial, sigma[i], seed[i])
        plt.savefig(('sigma=%.1f_deriv.png' % sigma[i]))
        plt.cla()