import numpy as np
import matplotlib.pyplot as plt
import cmath
import math

np.random.seed(1)
n_samples = 256
a1 = 0.5
a2 = -0.6
a3 = 0.3
a4 = -0.4
a5 = 0.2
x = n = np.random.normal(size=n_samples)
for t in range(5, n_samples):
    x[t] = a1 * x[t - 1] + a2 * x[t - 2] + a3 * x[t - 3] + a4 * x[t - 4] + a5 * x[t - 5] + n[t]


def periodogramm(inputt, omega, samples):
    const = 1 / (2 * math.pi * samples)
    expo = np.zeros([1, samples], dtype=np.complex)
    for i in range(0, 256):
        expo[0, i] = cmath.exp(0 - 1j * i * omega)
    return const * (abs(np.sum(inputt * expo)) ** 2)


w = np.linspace(0, math.pi, n_samples, endpoint=True)
Pxx_Density = np.zeros([n_samples, ])
for i1 in range(0, n_samples):
    Pxx_Density[i1] = periodogramm(x, w[i1], n_samples)


def kernel_function(arg):
    if abs(arg) <= 1:
        return 0.75 * (1 - arg ** 2)
    else:
        return 0


def spectral_density_estimator(h, powerspec, samples, freqrange, freqK):
    const = 1 / (h * samples)
    kernel = np.zeros([samples, ])
    for i in range(0, samples):
        kernel[i] = kernel_function((freqrange[freqK] -freqrange[i]) / h)
    summ = sum(const * kernel * powerspec)
    return summ


SpectralDensityEstimate = np.zeros([n_samples, ])
h = 0.05

for p in range(0, n_samples):
    SpectralDensityEstimate[p] = spectral_density_estimator(h, Pxx_Density, n_samples, w, p)

plt.subplot(3, 1, 1)
plt.plot(x)
plt.ylabel('AR(5) Time Series')
plt.xlabel('time T, 0<=T<=256')
plt.subplot(3, 1, 2)
plt.plot(w, Pxx_Density)
plt.xlabel('frequency W, 0<=W<=3.14(pi)')
plt.ylabel('Periodogram Ixx(W)')
plt.subplot(3, 1, 3)
plt.plot(w, SpectralDensityEstimate)
plt.xlabel('frequency W, 0<=W<=3.14(pi)')
plt.ylabel('PSD Estimate')
plt.subplots_adjust(hspace=0.5)
plt.show()