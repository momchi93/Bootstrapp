import numpy as np
T = 256
N = int(np.floor(T/2))
w = 2 * np.pi * np.arange(N+1) / T
omega = np.linspace(0, np.pi, (N+1)/T)
print(N)
print(w)
print(omega)