import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from one_spin_gif import one_spin_gif

class SOT(one_spin_gif):
    def __init__(self,alpha,gamma,B,S0,t,t_eval,spin_flow,Kx,Ky,Kz,beta,start,stop):
        super().__init__(alpha,gamma,B,S0,t,t_eval,plotB)
        self.spin_flow = spin_flow
        self.Kx = Kx
        self.Ky = Ky
        self.Kz = Kz
        self.beta = beta
        self.start = start
        self.stop = stop

    def func_S(self,t,S):  # 関数を設定
        sc_torque = []
        Snorm = np.linalg.norm(S)
        if t>= self.start and t <= self.stop:
            B_f = [self.B[0] + self.beta * self.spin_flow[0] + self.Kx * S[0]/(Snorm * Snorm), self.B[1] +self.beta * self.spin_flow[1] + self.Ky * S[1]/(Snorm * Snorm),
                    self.B[2] + self.beta * self.spin_flow[2] + self.Kz * S[2]/(Snorm * Snorm)]
            B_d = [self.B[0] + self.Kx * S[0]/(Snorm * Snorm), self.B[1] + self.Ky * S[1]/(Snorm * Snorm),
                    self.B[2] + self.Kz * S[2]/(Snorm * Snorm)]
            sc_torque = [self.gamma * (S[1] * (self.spin_flow[0] * S[1] - self.spin_flow[1] * S[0]) - S[2] * (
                        self.spin_flow[2] * S[0] - self.spin_flow[0] * S[2])), self.gamma * (
                                     S[2] * (self.spin_flow[1] * S[2] - self.spin_flow[2] * S[1]) - S[0] * (
                                         self.spin_flow[0] * S[1] - self.spin_flow[1] * S[0])), self.gamma * (
                                     S[0] * (self.spin_flow[2] * S[0] - self.spin_flow[0] * S[2]) - S[1] * (
                                         self.spin_flow[1] * S[2] - self.spin_flow[2] * S[1]))]
        else:
            B_f = [self.B[0] + self.Kx * S[0]/(Snorm * Snorm), self.B[1] + self.Ky * S[1]/(Snorm * Snorm), self.B[2] + self.Kz * S[2]/(Snorm * Snorm)]
            B_d = [self.B[0] + self.Kx * S[0]/(Snorm * Snorm), self.B[1] + self.Ky * S[1]/(Snorm * Snorm),
                    self.B[2] + self.Kz * S[2]/(Snorm * Snorm)]
            sc_torque = [0,0,0]

        Snorm = np.linalg.norm(S)


        dSxdt = - self.gamma * (B_f[2] * S[1] - B_f[1] * S[2])  - sc_torque[0] - (self.alpha / Snorm) * (
                    S[1] * (self.gamma * (B_d[1] * S[0] - B_d[0] * S[1])) - S[2] * (
                        self.gamma * (B_d[0] * S[2] - B_d[2] * S[0]) ))
        dSydt = - self.gamma * (B_f[0] * S[2] - B_f[2]  * S[0])  - sc_torque[1] - (self.alpha / Snorm) * (
                    S[2] * (self.gamma * (B_d[2] * S[1] - B_d[1] * S[2]) ) - S[0] * (
                        self.gamma * (B_d[1] * S[0] - B_d[0] * S[1]) ))
        dSzdt = - self.gamma * (B_f[1]  * S[0] - B_f[0] * S[1]) - sc_torque[2] - (self.alpha / Snorm) * (
                    S[0] * (self.gamma * (B_d[0] * S[2] - B_d[2] * S[0]) ) - S[1] * (
                        self.gamma * (B_d[2] * S[1] - B_d[1] * S[2])))
        dSdt = [dSxdt, dSydt, dSzdt]
        #print(dSdt)

        return dSdt

    def make_gif(self):
        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection="3d"))

        self.Sol = sc.integrate.solve_ivp(self.func_S, self.t, self.S0, t_eval=self.t_eval)
        self.S = self.Sol.y
        print(self.S)

        self.quiveraa = self.ax.quiver(*self.get_spin_vec(0))

        r  = np.linalg.norm(self.S0) # 半径を指定
        theta_1_0 = np.linspace(0, np.pi * 2, 100)  # θ_1は&#91;0,π/2]の値をとる
        theta_2_0 = np.linspace(0, np.pi, 100)  # θ_2は&#91;0,π/2]の値をとる
        theta_1, theta_2 = np.meshgrid(theta_1_0, theta_2_0)  # ２次元配列に変換
        x = np.cos(theta_2) * np.sin(theta_1) * r  # xの極座標表示
        y = np.sin(theta_2) * np.sin(theta_1) * r  # yの極座標表示
        z = np.cos(theta_1) * r  # zの極座標表示

        self.ax.plot_surface(x, y, z, alpha=0.2)  # 球を３次元空間に表示

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.text(0.12, 0.15, -0.3, '-B',
                size=20, )

        self.ax.view_init(elev=20, azim=20)
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_zlim(-1.2, 1.2)
        self.ax.set_aspect('equal')

        ani = FuncAnimation(self.fig, self.update, frames=len(self.Sol.t), interval=1)
        #ani.save("SOT_Type_X.gif",writer='imagemagick')
        plt.show()


if __name__ == '__main__':
    S0 = [0.968, 0, 0.199]

    t = [0, 200]  # t(時間)が0〜100まで動き、その時のfを求める。
    t_eval = np.linspace(*t, 2000)

    plotB = [[0, 0, -1.2], [0, 0, 2.4]]

    mu_0 = 1.2
    gamma = 0.028
    h_div_2e = [0.329, -15]
    sta_M = [1.4, 0]  # 飽和磁化(T)で入れる
    theta = [2.5, -1]
    j = [33.6, 11]
    d = [1.48, -9]
    Hsn = h_div_2e[0] * theta[0] * j[0] / (sta_M[0] * d[0])
    Hso = h_div_2e[1] + theta[1] + j[1] - (sta_M[1] + d[1])
    Hs = Hsn * (10 ** Hso) * 1000 * (mu_0 / 1200000) * mu_0  # 最後の1000はmTにするため
    print(Hs)

    spin = SOT(0.02, gamma, [0, 0, mu_0 * 17.5], S0, t, t_eval, [0, Hs, 0],  mu_0 * 4,0, -mu_0 * 83, -2, 0, 10)
    spin.make_gif()
