import matplotlib.pyplot as plt
import numpy as np
# t von Z, from t[0] to t[49] t<0,for t[50] t=0, 51-101 t>0;
t=101
#h for Model 1
h1=0.05
#list of t times the value of h1
h=[h1]*t
#h0
h[int((t-1)/2)]=1
#positive constant
sigma=1
#X is a real-valued,discrete-time,strictly stationary univariate process
X=[]
i=0
while i<=100:
    E= np.random.normal(size=t)
    x= sigma*sum(E*h)
    X.append(x)
    i+=1
plt.plot(X)
plt.show()