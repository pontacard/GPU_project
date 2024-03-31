import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def func_S(S,t):  # 関数を設定
    B = [0, np.sin(t), -4]
    Snorm = np.linalg.norm(S)
    dSxdt = - (-0.05) * (B[1] * S[2] - B[2] * S[1]) - (-0.01/Snorm) * (S[1] * (S[0] * B[1] - S[1] * B[0]) - S[2]* (S[2] * B[0] - S[0] * B[2]))
    dSydt = - (-0.05) * (B[2] * S[0] - B[0] * S[2]) - (-0.01/Snorm) * (S[2] * (S[1] * B[2] - S[2] * B[1]) - S[0]* (S[0] * B[1] - S[1] * B[0]))
    dSzdt = - (-0.05) * (B[0] * S[1] - B[1] * S[0]) - (-0.01/Snorm) * (S[0] * (S[2] * B[0] - S[0] * B[2]) - S[1]* (S[1] * B[2] - S[2] * B[1]))
    dSdt = [dSxdt, dSydt, dSzdt]

    return dSdt

fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

S0 = [0, 0.866, 0.5]

t = np.linspace(0, 1500, 9000)  # t(時間)が0〜100まで動き、その時のfを求める。

S = sc.integrate.odeint(func_S, S0, t)


def get_spin_vec(t):
    Ox,Oy,Oz = 0,0,0
    x = S[t][0]
    y = S[t][1]
    z = S[t][2]
    ax.plot(x, y, z, marker='o', markersize=2.5, color='b')
    return Ox, Oy, Oz, x, y, z




quiver = ax.quiver(*get_spin_vec(0), linewidth = 3)

ax.view_init(elev=20,azim=20)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)
ax.set_aspect('equal')

def update(t):
    global quiver
    quiver.remove()
    quiver = ax.quiver(*get_spin_vec(t), arrow_length_ratio=0.18,linewidth = 5)

r = 1 # 半径を指定
theta_1_0 = np.linspace(0, np.pi * 2, 100) # θ_1は&#91;0,π/2]の値をとる
theta_2_0 = np.linspace(0, np.pi, 100) # θ_2は&#91;0,π/2]の値をとる
theta_1, theta_2 = np.meshgrid(theta_1_0, theta_2_0) # ２次元配列に変換
x = np.cos(theta_2)*np.sin(theta_1) * r # xの極座標表示
y = np.sin(theta_2)*np.sin(theta_1) * r # yの極座標表示
z = np.cos(theta_1) * r # zの極座標表示

ax.plot_surface(x,y,z, alpha = 0.2) # 球を３次元空間に表示

ax.quiver(0,0,-1.2,0,0,2.4, color = 'k',arrow_length_ratio=0.18, linewidth = 3 , label = 'B')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.text(0.12, 0.15, -0.3, '-B',
            size=20,)




ani = FuncAnimation(fig, update, frames= range(1500), interval=1)
#ani.save("spin.gif",writer='imagemagick')
plt.show()


