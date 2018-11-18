import statsmodels.tsa.arima_process as a
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as s

np.random.seed(1)
ar1 = np.array([1, -0.5, 0.6, -0.3, 0.4, -0.2])
ma1 = np.array([1])
AR_object1 = a.ArmaProcess(ar1, ma1)
simulated_data_1 = AR_object1.generate_sample(nsample=256)  # type: AR_object1
plt.subplot(2, 1, 1)
plt.plot(simulated_data_1)
plt.subplot(2, 1, 2)
f, Pxx_den = s.periodogram(simulated_data_1) #I
plt.plot(f, Pxx_den)
#plt.ylim([1e-7, 1e2])
plt.show()
