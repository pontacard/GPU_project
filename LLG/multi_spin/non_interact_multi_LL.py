import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation

n = 3

def S(S,t):  # 関数を設定
    dSdt = np.empty(0)
    B = [0, 0, 5]
    for i in range(n):
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

S = np.reshape(y,(-1,n,n))

print(S)

def get_spin_vec(t,i):
    Ox,Oy,Oz = i,0,0
    x = S[t][i][0]
    y = S[t][i][1]
    z = S[t][i][2]
    return Ox, Oy, Oz, x, y, z

quiver_dic = dict()
for i in range(n):
    quiver_dic[i] = ax.quiver(*get_spin_vec(0,i))

ax.set_xlim(-2, 7)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

def update(t):
    global quiver_dic
    for i in range(n):
        quiver_dic[i].remove()
        quiver_dic[i] = ax.quiver(*get_spin_vec(t,i))

ani = FuncAnimation(fig, update, frames= range(1000), interval=1)
#ani.save("multi_spin.gif",writer='imagemagick')
plt.show()