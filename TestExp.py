import numpy as np
import cmath
import math
np.random.seed(1)
n_samples = 5
expo1 = np.zeros([1, n_samples], dtype=np.complex)
for i in range(0, n_samples):
    expo1[0, i] = cmath.exp(0-1j*i)
print(expo1)
x = np.random.normal(size=n_samples)
w = np.linspace(0, math.pi, n_samples, endpoint=True)
Pxx_Density = np.empty([1, n_samples])
print(w)
print(Pxx_Density)
