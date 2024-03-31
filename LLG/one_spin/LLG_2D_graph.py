import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def func_S(S,t):  # 関数を設定
    B = [0, 0, -4]
    Snorm = np.linalg.norm(S)
    dSxdt = - (-0.1) * (B[1] * S[2] - B[2] * S[1]) - (-0.005/Snorm) * (S[1] * (S[0] * B[1] - S[1] * B[0]) - S[2]* (S[2] * B[0] - S[0] * B[2]))
    dSydt = - (-0.1) * (B[2] * S[0] - B[0] * S[2]) - (-0.005/Snorm) * (S[2] * (S[1] * B[2] - S[2] * B[1]) - S[0]* (S[0] * B[1] - S[1] * B[0]))
    dSzdt = - (-0.1) * (B[0] * S[1] - B[1] * S[0]) - (-0.005/Snorm) * (S[0] * (S[2] * B[0] - S[0] * B[2]) - S[1]* (S[1] * B[2] - S[2] * B[1]))
    dSdt = [dSxdt, dSydt, dSzdt]

    return dSdt

S0 = [0, 0.7071, 0.7071]

t = np.linspace(0, 250, 250)  # t(時間)が0〜100まで動き、その時のfを求める。

S = sc.integrate.odeint(func_S, S0, t)

print(S)

def get_graph(S):
    x = S[:,0]
    y = S[:,1]
    z = S[:,2]
    S = np.square(x*x + y*y + z*z)
    print(S)
    #plt.plot(t, S, label = "|S|", ls = "--")
    #plt.savefig("reverse_|S|.png")
    #plt.plot(t,x)
    plt.plot(t, x, label="Sx")
    #plt.savefig("reverse_x.png")
    #plt.show()
    plt.plot(t, y, label = "Sy")
    #plt.savefig("reverse_y.png")
    #plt.show()
    plt.plot(t, z,  label = "Sz")

    plt.xlabel("time")
    #plt.axhline(color = 'k')
    plt.legend()

    plt.savefig("2d_graph1.png")
    plt.show()


def get_spin_vec(t):
    Ox,Oy,Oz = 0,0,0
    x = S[t][0]
    y = S[t][1]
    z = S[t][2]
    return Ox, Oy, Oz, x, y, z

get_graph(S)

