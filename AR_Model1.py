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


def periodogramm(imput, omega):
    const = 1 / (2 * math.pi * n_samples)
    expo = np.zeros([1, n_samples], dtype=np.complex)
    for i in range(0, 256):
        expo[0, i] = cmath.exp(0 - 1j * i * omega)
    return const * (abs(np.sum(imput * expo)) ** 2)


w = np.linspace(0, math.pi, n_samples, endpoint=True)
Pxx_Density = np.zeros([1, n_samples])
for i1 in range(0, n_samples):
    Pxx_Density[0, i1] = periodogramm(x, w[i1])
Pxx_Density = Pxx_Density.reshape(n_samples, )


def kernel_funct(arg):
    if abs(arg) <= 1:
        return 0.75 * (1 - arg ** 2)
    else:
        return 0



plt.subplot(2, 1, 1)
plt.plot(x)
plt.subplot(2, 1, 2)
plt.plot(w, Pxx_Density)
plt.show()
