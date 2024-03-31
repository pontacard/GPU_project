import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from one_spin.STO import STO_spin

class STO_gif(STO_spin):
    def __init__(self, alpha, gamma, B, S0, t, t_eval, H_Amp, omega, theta, Kx, Ky, Kz, beta, start, stop,STO_ac_effH, STO_dc_effH, lambdaa, eta):
        super().__init__(alpha, gamma, B, S0, t, t_eval, H_Amp, omega, theta, Kx, Ky, Kz, beta, start, stop,STO_ac_effH, STO_dc_effH, lambdaa, eta)

    def get_spin_vec(self,t):
        Ox, Oy, Oz = 0, 0, 0
        x = self.S[0][t]
        y = self.S[1][t]
        z = self.S[2][t]
        self.ax.plot(x, y, z, marker='o', markersize=2.5, color='b')
        return Ox, Oy, Oz, x, y, z

    def make_gif(self):
        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection="3d"))

        self.Sol = sc.integrate.solve_ivp(self.func_S, self.t, self.S0, t_eval=self.t_eval)
        self.S = self.Sol.y

        self.quiveraa = self.ax.quiver(*self.get_spin_vec(0))

        r  = np.linalg.norm(self.S0) # 半径を指定
        theta_1_0 = np.linspace(0, np.pi * 2, 100)  # θ_1は&#91;0,π/2]の値をとる
        theta_2_0 = np.linspace(0, np.pi, 100)  # θ_2は&#91;0,π/2]の値をとる
        theta_1, theta_2 = np.meshgrid(theta_1_0, theta_2_0)  # ２次元配列に変換
        x = np.cos(theta_2) * np.sin(theta_1) * r  # xの極座標表示
        y = np.sin(theta_2) * np.sin(theta_1) * r  # yの極座標表示
        z = np.cos(theta_1) * r  # zの極座標表示

        self.ax.plot_surface(x, y, z, alpha=0.2)  # 球を３次元空間に表示

        #self.ax.quiver(self.plotB[0][0], self.plotB[0][1], self.plotB[0][2], self.plotB[1][0], self.plotB[1][1], self.plotB[1][2], color='k', arrow_length_ratio=0.18, linewidth=3, label='B')
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.text(0.12, 0.15, -0.3, "B",size=20, )

        self.ax.view_init(elev=20, azim=20)
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_zlim(-1.2, 1.2)
        self.ax.set_aspect('equal')
        #self.ax.view_init(azim=1,elev=88)




        ani = FuncAnimation(self.fig, self.update, frames=len(self.Sol.t), interval=1)
        #ani.save(f"STO_{self.omega[1]}GHz_{self.STO_ac_effH[1]}Adiv^2.gif", fps = 40)
        #ani.save('pillow_imagedraw.gif', duration=40, loop=0)
        plt.show()

if __name__ == '__main__':
    S0 = [1, 0, 0]

    t = [0, 20]  # t(時間)が0〜100まで動き、その時のfを求める。
    t_eval = np.linspace(*t, 5000)

    plotB = [[0, 0, -1.2], [0, 0, 2.4]]

    gamma = 0.17
    mu_0 = 1.26
    mu_h_div_2e = [0.413563, -21]
    sta_M = [1.824, 0]  # 飽和磁化(T)で入れる
    jac = [4.1 ,11]
    #jac = [0.05, 11]
    d = [2, -9]
    Hacn = mu_h_div_2e[0] * jac[0] / (sta_M[0] * d[0])
    Haco = mu_h_div_2e[1] + jac[1] - (sta_M[1] + d[1])
    Hac = Hacn * (10 ** Haco) * 1000 / gamma  # 最後の1000はmTにするため
    print(Hac)

    jdc = [2.2 , 10]
    Hdcn = mu_h_div_2e[0] * jdc[0] / (sta_M[0] * d[0])
    Hdco = mu_h_div_2e[1] + jdc[1] - (sta_M[1] + d[1])
    Hdc = Hdcn * (10 ** Hdco) * 1000 / gamma  # 最後の1000はmTにするため

    Kx = 0
    Ky = 0
    Kz = 1481 - 1448

    spin = STO_gif(0.005, gamma, [0, 0, mu_0 * 159], S0, t, t_eval, [0, 0, 0], [0, 5.5, 0], [0, 0, 0], Kx, Ky,
                    mu_0 * Kz, 0, 0, 20, [0, Hac, 0], [0, Hdc, 0], 0.288, 0.537)
    spin.make_gif()
