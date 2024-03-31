import scipy as sc
import numpy as np
import matplotlib.pyplot as plt

K = 1
M = 1
h =0.1
gamma = 28
H0 = 0
H1= -1/5.6
H2 = -1/2.8

mu = 1
theta = np.linspace(-1.5,4.7, 300)

V0 = K/2 * M*M * np.sin(theta) * np.sin(theta) -h * gamma * mu * H0 *M* (np.cos(theta) - 1)
V1 = K/2 * M*M * np.sin(theta) * np.sin(theta) -h * gamma * mu * H1 *M* (np.cos(theta) - 1 )
V2 = K/2 * M*M * np.sin(theta) * np.sin(theta) -h * gamma * mu * H2 *M* (np.cos(theta) - 1)
plt.plot(theta,V0)
#plt.plot(theta,V1,label = "H = Hc/2")
#plt.plot(theta,V2,label = "H = Hc")
#plt.legend()
plt.xlabel("Sz")
plt.ylabel("U")
plt.xticks([])
plt.yticks([])
plt.ylim(-0.01,0.7)
plt.grid(axis='y')
#plt.savefig("V(θ)_2023.pdf")

plt.show()