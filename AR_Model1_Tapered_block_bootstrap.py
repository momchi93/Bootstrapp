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
    for i3 in range(N + 1):
        Pxx_Density[i3] = (1 / samples) * np.abs(np.sum(inputt * np.exp(-1j * omega[i3] * t))) ** 2
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


def draw_block():
    x_block = np.zeros(b)
    i = np.random.choice(set_for_i)
    for j in range(b):
        x_block[j] = tp_w[j] * const * x_centered[i + j]
    return x_block


def get_resample():
    X_resamples_list = []
    for p in range(k):
        block = draw_block()
        X_resamples_list.append(block)
    X_resamples = np.concatenate(np.array(X_resamples_list), axis=None)
    return X_resamples


n_samples = 256
n = int(np.floor(n_samples / 2))
h = g = 0.05
b = 38
b_samples = 30
ar = np.array([0.5, -0.6, 0.3, -0.4, 0.2])
set_for_i = np.arange(1, n_samples - b + 2)  # integers representing start of a block
w1 = np.arange(1, 17) / 16
w2 = np.ones(6)
w3 = np.flip(w1, axis=None)
tp_w = np.concatenate((w1, w2, w3), axis=None)  # tapering window
tp_w_2 = np.sqrt(np.sum(tp_w))  # Sum of tapering windows, squared
const = np.sqrt(b) / tp_w_2

# Generate AR(5) Series
x = generate_ar_process(n_samples, ar)

# Step 1 Center
x_centered = x - x.mean()

# Step 2 Initial Estimate
Ixx_density_centered = periodogramm(x_centered, n_samples)
Cxx_estimate_centered = spectral_density_estimator(h, Ixx_density_centered, n_samples)

# Step 3 Set the bootstrap parameters
k = int(np.floor(n_samples / b))
l = k * b
l_half = int(l / 2)

# Step 6 Repeat Step 4 & 5
Cxx_estimate_centered_b = np.zeros([b_samples, l_half + 1])

for b in range(b_samples):
    # Step 4 Draw Bootstrap Resamples
    X_resample = get_resample()
    # Step 5 Construct Bootstrap Estimate
    X_resamples_centered = X_resample - np.mean(X_resample)
    #Ixx_density_centered_b = periodogramm(X_resamples_centered, l)
    #Ixx_density_centered_b[0] = 0
    #Cxx_estimate_centered_b[b, :] = spectral_density_estimator(h, Ixx_density_centered_b, l)

w = 2 * np.pi * np.arange(n + 1) / n_samples
w1 = 2 * np.pi * np.arange(l_half + 1) / l
#plt.plot(w, Cxx_estimate_centered)
#plt.plot(w1, Cxx_estimate_centered_b)
#plt.show()

print('finish')
