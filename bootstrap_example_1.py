
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_process import ArmaProcess


# Function Defintions


def generate_ar_process(N, ar):
    ar_process = ArmaProcess(np.r_[1, -ar], np.array([1]))
    X = ar_process.generate_sample(N)
    return X


def get_I_XX(X, N):
    T = X.size
    t = np.arange(T)

    w = 2 * np.pi * np.arange(N+1) / T

    I_XX = np.zeros(N+1)
    for k in range(N+1):
        I_XX[k] = (1/T) * np.abs(np.sum(X * np.exp(-1j * w[k] * t))) ** 2

    return I_XX


def kernel(x):
    mask = np.abs(x) < 1
    K = np.zeros(x.size)
    K[mask] = 0.75*(1-x[mask]**2)

    return K


def get_C_XX_hat(I_XX, T, h):
    N = I_XX.size-1

    w = 2 * np.pi * np.arange(N+1) / T

    C_XX_hat = np.zeros(N+1)
    for k in range(N+1):
        C_XX_hat[k] = (2 * np.pi) * np.sum(kernel((w[k]-w)/h) * I_XX) / (h*T)

    return C_XX_hat


# Bootstrap Example


# set parameter
np.random.seed(123)
T = 256
N = int(np.floor(T/2))
h = 0.05
g = 0.05
BSR = 100
alpha = 0.05
ar = np.array([0.5, -0.6, 0.3, -0.4, 0.2])

# generate sample
X = generate_ar_process(T, ar)

# calculate periodogram and smoothed periodogram
I_XX = get_I_XX(X, N)
C_XX_hat = get_C_XX_hat(I_XX, T, h)

# calculate scaled residuals
eps = I_XX/C_XX_hat
eps_mean = np.mean(eps)
eps_scaled = eps/eps_mean

# draw BSR boostrap resamples and calculate smoothed periodgrams
C_XX_hat_BS = np.zeros([BSR, N+1])

for b in range(BSR):
    eps_star = np.random.choice(eps, eps.size)
    I_XX_star = C_XX_hat*eps_star
    I_XX_star[0] = 0
    C_XX_hat_BS[b, :] = get_C_XX_hat(I_XX_star, T, g)

# get confidence bounds
upper_index = int(np.ceil((BSR-1)*(1-alpha)))
lower_index = int(np.floor((BSR-1)*alpha))

C_XX_hat_BS.sort(axis=0)

C_XX_hat_upper = C_XX_hat_BS[upper_index, :]
C_XX_hat_lower = C_XX_hat_BS[lower_index, :]

# plot estimate and bounds
w = 2*np.pi*np.arange(N+1)/T
plt.plot(w, C_XX_hat)
plt.plot(w, C_XX_hat_upper)
plt.plot(w, C_XX_hat_lower)
plt.show()