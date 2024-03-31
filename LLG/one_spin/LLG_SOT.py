import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class A_spin():
    def __init__(self,alpha,gamma,B,S0,t):
        self.alpha = alpha  #緩和項をどれくらい大きく入れるか
        self.gamma = gamma  #LLg方程式のγ
        self.B = B          #外部磁場(tで時間変化させてもいいよ)
        self.S0 = S0        #初期のスピンの向き
        self.t = t          #どれくらいの時間でやるか
        self.S = []
        self.ax = 0         #pltのオブジェクト
        self.fig = 0        #pltのオブジェクト
        self.quiveraa = 0

    def func_S(self,S,t):  # 関数を設定
        B = self.B
        Snorm = np.linalg.norm(S)
        dSxdt = - self.gamma * (B[1] * S[2] - B[2] * S[1]) - (self.gamma * self.alpha/Snorm) * (S[1] * (S[0] * B[1] - S[1] * B[0]) - S[2]* (S[2] * B[0] - S[0] * B[2]))
        dSydt = - self.gamma * (B[2] * S[0] - B[0] * S[2]) - (self.gamma * self.alpha/Snorm) * (S[2] * (S[1] * B[2] - S[2] * B[1]) - S[0]* (S[0] * B[1] - S[1] * B[0]))
        dSzdt = - self.gamma * (B[0] * S[1] - B[1] * S[0]) - (self.gamma * self.alpha/Snorm) * (S[0] * (S[2] * B[0] - S[0] * B[2]) - S[1]* (S[1] * B[2] - S[2] * B[1]))
        dSdt = [dSxdt, dSydt, dSzdt]

        return dSdt

    def get_spin_vec(self,t):
        Ox,Oy,Oz = 0,0,0
        x = self.S[t][0]
        y = self.S[t][1]
        z = self.S[t][2]
        return Ox, Oy, Oz, x, y, z


    def update(self,t):
        self.quiveraa.remove()
        self.quiveraa = self.ax.quiver(*self.get_spin_vec(t))

    def doit(self):
        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection="3d"))

        self.S = sc.integrate.odeint(self.func_S, self.S0, self.t)


        self.quiveraa = self.ax.quiver(*self.get_spin_vec(0))

        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(-2, 2)

        ani = FuncAnimation(self.fig, self.update, frames=len(t), interval=1)
        # ani.save("reverse_spin.gif",writer='imagemagick')
        plt.show()

if __name__ == '__main__':
    S0 = [0.1, 0, -0.9]

    t = np.linspace(0, 1500, 1500)  # t(時間)が0〜100まで動き、その時のfを求める。

    spin = A_spin(0.01,0.1,[0,0,5],S0,t)
    spin.doit()
