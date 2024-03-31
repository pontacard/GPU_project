import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation

def S(S,t):  # 関数を設定
    dSdt = np.empty(0)
    B = [0, 0, 5]
    for i in range(3):
        dSixdt = - (-0.1) * (B[1] * S[3*i + 2] - B[2] * S[3*i +1])
        dSiydt = - (-0.1) * (B[2] * S[3*i +0] - B[0] * S[3*i +2])
        dSizdt = - (-0.1) * (B[0] * S[3*i +1] - B[1] * S[3*i +0])
        dSdt = np.append(dSdt,[dSixdt, dSiydt, dSizdt])
    dSdt = np.reshape(dSdt,(1,-1))

    return dSdt[0]


S0 = np.array([[1,0,0],[1,0,1],[1,0,0]])

S0 = np.reshape(S0,(1,-1))


t = np.linspace(0, 100, 1000)   # t(時間)が0〜3まで動き、その時のfを求める。

y = sc.integrate.odeint(S, S0[0], t)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

frames = []

S = np.reshape(y,(-1,3,3))

print(S)

def get_spin1_vec(t):
    Ox,Oy,Oz = 0,0,0
    x = S[t][0][0]
    y = S[t][0][1]
    z = S[t][0][2]
    return Ox, Oy, Oz, x, y, z

def get_spin2_vec(t):
    Ox,Oy,Oz = 2,0,0
    x = S[t][1][0]
    y = S[t][1][1]
    z = S[t][1][2]
    return Ox, Oy, Oz, x, y, z


quiver = ax.quiver(*get_spin1_vec(0))
quiver2 = ax.quiver(*get_spin2_vec(0))

ax.set_xlim(-2, 7)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

def update(t):
    global quiver, quiver2
    quiver.remove()
    quiver = ax.quiver(*get_spin1_vec(t))
    quiver2.remove()
    quiver2 = ax.quiver(*get_spin2_vec(t))

ani = FuncAnimation(fig, update, frames= range(1000), interval=1)
#ani.save("multi_spin.gif",writer='imagemagick')
plt.show()

