import numpy as np
import matplotlib.pyplot as plt
import time


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
        SpectralDensityEstimate[i1] = (2 * np.pi) * sum((kernel * powerspec) / (h1 * samples))
    return SpectralDensityEstimate


def draw_residuals(residualss, samples):
    independent_bootstrap_residuals = np.zeros(samples)
    for i3 in range(samples):
        independent_bootstrap_residuals[i3] = np.random.choice(residualss)
    return independent_bootstrap_residuals


# Step 5 - Bootstrap Estimate
def bootstrap_estimate(res, g, sde_centered, samples):
    residualz = draw_residuals(res, samples)  # e
    pxxdensity = sde_centered * residualz
    pxxdensity[0] = 0
    bootstrap_spectral_density_estimate = spectral_density_estimator(g, pxxdensity, samples)
    return bootstrap_spectral_density_estimate


np.random.seed(123)
start_time = time.time()
n_samples = 256
n = int((n_samples / 2) + 1)
a1 = 0.5
a2 = -0.6
a3 = 0.3
a4 = -0.4
a5 = 0.2
alpha = 0.05
h = g = 0.05
x = noise = np.random.normal(size=n_samples)
w = (2 * np.pi * np.arange(n)) / n_samples
for t in range(5, n_samples):
    x[t] = a1 * x[t - 1] + a2 * x[t - 2] + a3 * x[t - 3] + a4 * x[t - 4] + a5 * x[t - 5] + noise[t]

# Step 1 Center

x_centered = x - x.mean()

# Step 2 Initial Estimate

Ixx_density_centered = periodogramm(x_centered, n_samples)
Cxx_estimate_centered = spectral_density_estimator(h, Ixx_density_centered, n_samples)

# Step 3 Compute and Rescale Residuals

residuals = Ixx_density_centered / Cxx_estimate_centered
residuals_rescaled = residuals / np.mean(residuals)

# Step 4,5 & 6 Bootstrap Residuals and Estimate
b_samples = 100
b_sde = np.zeros([b_samples, n])
for c in range(b_samples):
    b_sde[c, :] = bootstrap_estimate(residuals_rescaled, g, Cxx_estimate_centered, n)

# Step 7 Confidence Interval Estimation

upper_index = int(np.ceil((b_samples - 1) * (1 - alpha)))
lower_index = int(np.floor((b_samples - 1) * alpha))

b_sde.sort(axis=0)

Cxx_estimate_centered_upper = b_sde[upper_index, :]
Cxx_estimate_centered_lower = b_sde[lower_index, :]

plt.plot(w, Cxx_estimate_centered, label='Spectral Density Estimate')
plt.plot(w, Cxx_estimate_centered_upper, label='upper bound')
plt.plot(w, Cxx_estimate_centered_lower, label='lower bound')
plt.legend(loc='best')
plt.show()

# test
b_samples = 6
Monte_Carlo_runs = 100
coverage_probability_list = []
for d in range(Monte_Carlo_runs):
    b_sde_Monte_Carlo = np.zeros([b_samples, n])
    for d1 in range(b_samples):
        b_sde_Monte_Carlo[d1, :] = bootstrap_estimate(residuals_rescaled, g, Cxx_estimate_centered, n)
    b_sde_Monte_Carlo.sort(axis=0)
    Cxx_estimate_centered_upper = b_sde_Monte_Carlo[b_samples - 1, :]
    Cxx_estimate_centered_lower = b_sde_Monte_Carlo[0, :]
    low_hits = np.count_nonzero(Cxx_estimate_centered_lower < Cxx_estimate_centered)
    up_hits = np.count_nonzero(Cxx_estimate_centered < Cxx_estimate_centered_upper)
    coverage_probability_list.append((low_hits + up_hits) / (2 * np.size(Cxx_estimate_centered_lower)))

coverage_probability = 100 * np.mean(np.array(coverage_probability_list))
print('Frequency Domain Residual-based Bootstrap')
print('bootstrap_samples = ' + str(b_samples) + ' , Monte_Carlo_runs = ' + str(Monte_Carlo_runs))
print('Coverage Probability = ' + str(coverage_probability) + '%')
elapsed_time = time.time() - start_time
print('elapsed_time for coverage_probability: ' + str(elapsed_time))
