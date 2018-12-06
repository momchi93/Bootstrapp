import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.arima_process
import time

def generate_ar_process(N, ar):
    ar_process = statsmodels.tsa.arima_process.ArmaProcess(np.r_[1, -ar], np.array([1]))
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


def draw_block(time_series, tap_wind, length):
    x_block = np.zeros(length)
    i = np.random.choice(set_for_i)
    for j in range(length):
        x_block[j] = const * tap_wind[j] * time_series[i + j]
    return x_block


def get_resample(time_series, tap_wind, length):
    x_resamples_list = []
    for p in range(k):
        block = draw_block(time_series, tap_wind, length)
        x_resamples_list.append(block)
    x_resamples = np.concatenate(np.array(x_resamples_list), axis=None)
    return x_resamples


def get_resample_full(time_series, tap_wind, length):
    x_resamples_list = []
    for p in range(k):
        block = draw_block(time_series, tap_wind, length)
        x_resamples_list.append(block)
    block_last = draw_block(time_series, tap_wind, n_samples - l)
    x_resamples_list.append(block_last)
    x_resamples = np.concatenate(np.array(x_resamples_list), axis=None)
    return x_resamples


np.random.seed(123)
start_time = time.time()
n_samples = 256  # length of time series
n = int(np.floor(n_samples / 2))  # half of time series
h = g = 0.05
b = 38  # block length
b_samples = 100  # bootstrap samples
ar = np.array([0.5, -0.6, 0.3, -0.4, 0.2])  # AR lag coefficient
set_for_i = np.arange(1, n_samples - b + 1)  # integers representing start of a block
w1 = np.arange(1, 17) / 16
w2 = np.ones(6)
w3 = np.flip(w1, axis=None)
tp_w = np.concatenate((w1, w2, w3), axis=None)  # tapering window
tp_w_2 = np.sqrt(np.sum(tp_w))  # sum of tapering windows, squared
const = np.sqrt(b) / tp_w_2  # constant for the draw_block method
alpha = 0.05  # confidence interval [alpha : 1 - alpha]

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
for b1 in range(b_samples):
    # Step 4 Draw Bootstrap Resamples
    X_resample = get_resample(x_centered, tp_w, b)
    # Step 5 Construct Bootstrap Estimatef
    X_resamples_centered = X_resample - np.mean(X_resample)
    Ixx_density_centered_b = periodogramm(X_resample, l)
    Ixx_density_centered_b[0] = 0
    Cxx_estimate_centered_b[b1, :] = spectral_density_estimator(h, Ixx_density_centered_b, l)

Cxx_estimate_centered_b_full = np.zeros([b_samples, n + 1])
for b2 in range(b_samples):
    # Step 4 Draw Bootstrap Resamples
    X_resample = get_resample_full(x_centered, tp_w, b)
    # Step 5 Construct Bootstrap Estimatef
    X_resamples_centered = X_resample - np.mean(X_resample)
    Ixx_density_centered_b = periodogramm(X_resample, n_samples)
    Ixx_density_centered_b[0] = 0
    Cxx_estimate_centered_b_full[b2, :] = spectral_density_estimator(h, Ixx_density_centered_b, n)
# Step 7 Confidence Interval Estimation

upper_index = int(np.ceil((b_samples - 1) * (1 - alpha)))
lower_index = int(np.floor((b_samples - 1) * alpha))

Cxx_estimate_centered_b.sort(axis=0)
Cxx_estimate_centered_b_full.sort(axis=0)

Cxx_estimate_centered_upper = Cxx_estimate_centered_b[upper_index, :]
Cxx_estimate_centered_lower = Cxx_estimate_centered_b[lower_index, :]
Cxx_estimate_centered_full_upper = Cxx_estimate_centered_b_full[upper_index, :]
Cxx_estimate_centered_full_lower = Cxx_estimate_centered_b_full[lower_index, :]

w = 2 * np.pi * np.arange(n + 1) / n_samples
w_l = 2 * np.pi * np.arange(l_half + 1) / l
Tapered_X1 = get_resample_full(x_centered, tp_w, b)
Tapered_X2 = get_resample_full(x_centered, tp_w, b)


plt.subplot(221)
plt.plot(w, Cxx_estimate_centered, label='Spectral Density Estimate')
plt.plot(w_l, Cxx_estimate_centered_upper, label='upper bound')
plt.plot(w_l, Cxx_estimate_centered_lower, label='lower bound')
plt.legend(loc='best')
plt.subplot(222)
plt.plot(w, Cxx_estimate_centered, label='Spectral Density Estimate')
plt.plot(w, Cxx_estimate_centered_full_upper, label='upper bound')
plt.plot(w, Cxx_estimate_centered_full_lower, label='lower bound')
plt.legend(loc='best')
plt.subplot(223)
plt.plot(Tapered_X1, label='Tapered_X1')
plt.plot(x_centered, label='Original_X')
plt.legend(loc='best')
plt.subplot(224)
plt.plot(Tapered_X2, label='Tapered_X2')
plt.plot(x_centered, label='Original_X')
plt.legend(loc='best')
#plt.show()

# test
b_samples = 10
upper_index = int(np.ceil((b_samples - 1) * (1 - alpha)))
lower_index = int(np.floor((b_samples - 1) * alpha))
coverage_probability_list = []
for d in range(100):
    Cxx_estimate_centered_b_full_Monte_Carlo = np.zeros([b_samples, n + 1])
    for b3 in range(b_samples):
        # Step 4 Draw Bootstrap Resamples
        X_resample = get_resample_full(x_centered, tp_w, b)
        # Step 5 Construct Bootstrap Estimatef
        X_resamples_centered = X_resample - np.mean(X_resample)
        Ixx_density_centered_b = periodogramm(X_resample, n_samples)
        Ixx_density_centered_b[0] = 0
        Cxx_estimate_centered_b_full_Monte_Carlo[b3, :] = spectral_density_estimator(h, Ixx_density_centered_b, n)
        # Step 7 Confidence Interval Estimation
        Cxx_estimate_centered_b_full_Monte_Carlo.sort(axis=0)
        Cxx_estimate_centered_upper = Cxx_estimate_centered_b_full_Monte_Carlo[upper_index, :]
        Cxx_estimate_centered_lower = Cxx_estimate_centered_b_full_Monte_Carlo[lower_index, :]
        low_hits = np.count_nonzero(Cxx_estimate_centered_lower < Cxx_estimate_centered)
        up_hits = np.count_nonzero(Cxx_estimate_centered < Cxx_estimate_centered_upper)
        coverage_probability_list.append((low_hits + up_hits) / (2 * np.size(Cxx_estimate_centered_lower)))

coverage_probability = 100 * np.mean(np.array(coverage_probability_list))
print('Coverage Probability = ' + str(coverage_probability) + '%')
elapsed_time = time.time() - start_time
print('elapsed_time for coverage_probability: ' + str(elapsed_time))
