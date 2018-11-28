import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess


def generate_ar_process(N, ar):
    ar_process = ArmaProcess(np.r_[1, -ar], np.array([1]))
    X = ar_process.generate_sample(N)
    return X


def periodogramm(inputt, samples):
    t = np.arange(samples)
    N = int(np.floor(samples / 2))
    omega = 2 * np.pi * np.arange(N + 1) / samples
    Pxx_Density = np.zeros(N + 1)
    for i1 in range(N + 1):
        Pxx_Density[i1] = (1 / samples) * np.abs(np.sum(inputt * np.exp(-1j * omega[i1] * t))) ** 2
    return Pxx_Density


# The Bartlett-Priestly Window: M.B.Priestly, Spectral Analysis and Time Series, page 444
def kernel_function(arg):
    if abs(arg) <= 1:
        return 0.75 * (1 - arg ** 2)
    else:
        return 0


def spectral_density_estimator(h1, powerspec, samples):
    N = powerspec.size - 1
    omega = 2 * np.pi * np.arange(N + 1) / samples
    SpectralDensityEstimate = np.zeros(N + 1)
    for i1 in range(N + 1):
        kernel = np.zeros(N + 1)
        for i2 in range(N + 1):
            kernel[i2] = kernel_function((omega[i1] - omega[i2]) / h1)
        SpectralDensityEstimate[i1] = sum((kernel * powerspec) / (h1 * samples))
    return SpectralDensityEstimate


n_samples = 256
n = int(np.floor(n_samples / 2))
h = g = 0.05
ar = np.array([0.5, -0.6, 0.3, -0.4, 0.2])

# Generate AR(5) Series
x = generate_ar_process(n_samples, ar)

# Step 1 Center
x_centered = x - x.mean()

# Step 2 Initial Estimate
Ixx_density_centered = periodogramm(x_centered, n_samples)
Cxx_estimate_centered = spectral_density_estimator(h, Ixx_density_centered, n_samples)

# Step 3 Set the bootstrap parameters
b = 38
k = int(np.floor(n_samples / b))


# Step 4 Draw Bootstrap Resamples
set_for_i = np.arange(1, n_samples-b+2)
i = np.zeros(k)
for p in range(k):
    i[p] = np.random.choice(set_for_i)
m = np.arange(k)

