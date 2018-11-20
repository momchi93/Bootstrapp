import numpy as np
import matplotlib.pyplot as plt
import cmath
import math


def periodogramm(inputt, omega, samples):
    const = 1 / (2 * math.pi * samples)
    expo = np.zeros([1, samples], dtype=np.complex)
    for i1 in range(0, samples):
        expo[0, i1] = cmath.exp(0 - 1j * i1 * omega)
    return const * (abs(np.sum(inputt * expo)) ** 2)


# The Bartlett-Priestly Window: M.B.Priestly, Spectral Analysis and Time Series, page 444
def kernel_function(arg):
    if abs(arg) <= 1:
        return 0.75 * (1 - arg ** 2)
    else:
        return 0


def spectral_density_estimator(h1, powerspec, samples, freqrange, freqK):
    const = 1 / (h1 * samples)
    kernel = np.zeros([samples, ])
    for i2 in range(0, samples):
        kernel[i2] = kernel_function((freqrange[freqK] - freqrange[i2]) / h1)
    summ = sum(const * kernel * powerspec)
    return summ


np.random.seed(123)
n_samples = 256
a1 = 0.5
a2 = -0.6
a3 = 0.3
a4 = -0.4
a5 = 0.2
x = n = np.random.normal(size=n_samples)
for t in range(5, n_samples):
    x[t] = a1 * x[t - 1] + a2 * x[t - 2] + a3 * x[t - 3] + a4 * x[t - 4] + a5 * x[t - 5] + n[t]

w = np.linspace(0, math.pi, n_samples, endpoint=True)
Pxx_Density = np.zeros([n_samples, ])
for i in range(0, n_samples):
    Pxx_Density[i] = periodogramm(x, w[i], n_samples)

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
# plt.show()

print('The residual based bootstrap residual')
# input('Press key to continue')

# Step 1 - Centering

x_centered = np.empty([n_samples, ])
x_mean = sum(x) / len(x)
for i in range(0, n_samples):
    x_centered[i] = x[i] - x_mean

# Step 2 - Initial Estimate

Pxx_Density_Centered = np.zeros([n_samples, ])
SpectralDensityEstimate_Centered = np.zeros([n_samples, ])
for i in range(0, n_samples):
    Pxx_Density_Centered[i] = periodogramm(x_centered, w[i], n_samples)
for p in range(0, n_samples):
    SpectralDensityEstimate_Centered[p] = spectral_density_estimator(h, Pxx_Density_Centered, n_samples, w, p)

# Step 3 - Compute and Rescale Residuals

residuals = residuals_rescaled = np.zeros([n_samples, ])
for i in range(0, n_samples):
    residuals[i] = Pxx_Density_Centered[i] / SpectralDensityEstimate_Centered[i]
residuals_mean = np.mean(residuals)

for i in range(0, n_samples):
    residuals_rescaled[i] = (residuals[i] / residuals_mean)


# Step 4 - Bootstrap Residuals


def draw_residuals(residualss, samples):
    independent_bootstrap_residuals = np.zeros([samples, ])
    for i3 in range(0, samples):
        independent_bootstrap_residuals[i3] = np.random.choice(residualss)
    return independent_bootstrap_residuals


print('SpectralDensityEstimate_Centered_0 = ' + str(SpectralDensityEstimate_Centered[0]))


# Step 5 - Bootstrap Estimate
def bootstrap_estimate(res, g, samples, freqrange):
    residualz = draw_residuals(res, samples)
    pxxdensity = SpectralDensityEstimate_Centered * residualz
    bootstrap_spectral_density_estimate = np.zeros([samples, ])
    for p1 in range(0, samples):
        bootstrap_spectral_density_estimate[p1] = spectral_density_estimator(g, pxxdensity, samples, freqrange, p1)
    return bootstrap_spectral_density_estimate


b_samples = 300
b_sde0 = np.empty([b_samples, ])
for i in range(b_samples):
    b_sde0[i] = bootstrap_estimate(residuals_rescaled, 0.05, n_samples, w)[0]
print('Mean = ' + str(np.mean(b_sde0)) + '  Varianz = ' + str(np.var(b_sde0)))
