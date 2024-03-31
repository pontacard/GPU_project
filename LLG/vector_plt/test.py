import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def func_S(S,t):  # 関数を設定
    B = [0, 0, 50]
    gamma = -0.1
    dSxdt = - gamma * (B[1] * S[2] - B[2] * S[1])
    dSydt = - gamma * (B[2] * S[0] - B[0] * S[2])
    dSzdt = - gamma * (B[0] * S[1] - B[1] * S[0])
    dSdt = [dSxdt, dSydt, dSzdt]

    return dSdt

fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))


S0 = [0.1, 0, 1]

t = np.linspace(0, 100, 1000)  # t(時間)が0〜100まで動き、その時のfを求める。

S = sc.integrate.odeint(func_S, S0, t)


def get_spin_vec(t):
    Ox,Oy,Oz = 0,0,0
    x = S[t][0]
    y = S[t][1]
    z = S[t][2]
    return Ox, Oy, Oz, x, y, z

quiver = ax.quiver(*get_spin_vec(0))

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

def update(t):
    global quiver
    quiver.remove()
    quiver = ax.quiver(*get_spin_vec(t))

ani = FuncAnimation(fig, update, frames= range(1000), interval=1)
#ani.save("multi_spin.gif",writer='imagemagick')
plt.show()