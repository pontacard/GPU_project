import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import ArtistAnimation

def S(S,t):  # 関数を設定
    B = [0, 0, 5]
    dSxdt = - (-0.1) * (B[1] * S[2] - B[2] * S[1])
    dSydt = - (-0.1) * (B[2] * S[0] - B[0] * S[2])
    dSzdt = - (-0.1) * (B[0] * S[1] - B[1] * S[0])
    dSdt = [dSxdt, dSydt, dSzdt]

    return dSdt


S0 = [1,0,0]

t = np.linspace(0, 100, 1000)   # t(時間)が0〜3まで動き、その時のfを求める。

y = sc.integrate.odeint(S, S0, t)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

frames = []
for dt in range(1000):
    S = y[dt]
    frame = ax.plot(S[0], S[1], S[2], marker='o', markersize=5, color='blue')
    frames.append(frame)

ani = ArtistAnimation(fig, frames, interval=1)

plt.show()