import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from macro_spin.tool import Tool
import random


class Thermal_spin(Tool):
    def __init__(self, alpha, gamma, B, S0, t, t_eval, Amp, omega, theta, Kx, Ky, Kz, beta,sigma,ther_dt, start, stop):
        super().__init__(t, t_eval)
        self.alpha = alpha
        self.gamma = gamma
        self.B = B
        self.S0 = S0
        self.Amp = Amp
        self.omega = omega
        self.Kx = Kx
        self.Ky = Ky
        self.Kz = Kz
        self.beta = beta
        self.start = start
        self.stop = stop
        self.theta = theta
        self.sigma = sigma
        self.ther_dt = ther_dt
        self.BT = np.array([random.gauss(0, self.sigma) , random.gauss(0, self.sigma),random.gauss(0, self.sigma)])
        self.t_check = 0
        self.t_before = 0


    def func_S(self, t, S):  # 関数を設定
        Snorm = np.linalg.norm(S)
        self.t_check = t // self.ther_dt #時間が細かい実数値をとるから、幅ther_dtで時間を離散化する
        BT_num = int(t // self.ther_dt)
        #print(self.rndBx,t // self.ther_dt)

        self.BT = [self.rndBx[BT_num], self.rndBy[BT_num],self.rndBz[BT_num]]
        #print(t,self.BT)
        #print(t,self.BT)
        Bk = np.array([self.Kx * S[0] / (Snorm * Snorm), self.Ky * S[1] / (Snorm * Snorm), self.Kz * S[2] / (Snorm * Snorm)])
        if t >= self.start and t <= self.stop:
            B = np.array(self.B) + self.BT+ Bk + np.array([self.Amp[0] * np.cos(self.omega[0] * t + self.theta[0]),
                                                  self.Amp[1] * np.sin(self.omega[1] * t + self.theta[1]),
                                                  self.Amp[2] * np.sin(self.omega[2] * t + self.theta[2])])
            # print(B)
        else:
            B = np.array(self.B) + self.BT + Bk

        Snorm = np.linalg.norm(S)
        dSxdt = - self.gamma * (S[1] * B[2] - S[2] * B[1]) - (self.gamma * self.alpha / Snorm) * (
                    S[1] * (S[0] * B[1] - S[1] * B[0]) - S[2] * (S[2] * B[0] - S[0] * B[2]))
        dSydt = - self.gamma * (S[2] * B[0] - S[0] * B[2]) - (self.gamma * self.alpha / Snorm) * (
                    S[2] * (S[1] * B[2] - S[2] * B[1]) - S[0] * (S[0] * B[1] - S[1] * B[0]))
        dSzdt = - self.gamma * (S[0] * B[1] - S[1] * B[0]) - (self.gamma * self.alpha / Snorm) * (
                    S[0] * (S[2] * B[0] - S[0] * B[2]) - S[1] * (S[1] * B[2] - S[2] * B[1]))
        dSdt = [dSxdt, dSydt, dSzdt]

        self.t_before = t // self.ther_dt #前の

        return dSdt

    def doit(self):
        rnd_B_size = self.t[1] //self.ther_dt + 1
        generator = np.random.default_rng()
        self.rndBx = generator.normal(loc=0, scale=self.sigma, size=rnd_B_size)
        self.rndBy = generator.normal(loc=0, scale=self.sigma, size=rnd_B_size)
        self.rndBz = generator.normal(loc=0, scale=self.sigma, size=rnd_B_size)

        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection="3d"))

        self.Sol = sc.integrate.solve_ivp(self.func_S,self.t, self.S0, t_eval=self.t_eval,atol=1e-12,rtol=1e-12)
        self.S = self.Sol.y


        self.quiveraa = self.ax.quiver(*self.get_spin_vec(0))

        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(-2, 2)

        ani = FuncAnimation(self.fig, self.update, frames=len(self.Sol.t), interval=100)
        # ani.save("reverse_spin.gif",writer='imagemagick')
        plt.show()

        return self.S

    def history(self):
        rnd_B_size = int(self.t[1] // self.ther_dt) + 1
        generator = np.random.default_rng()
        self.rndBx = generator.normal(loc=0, scale=self.sigma, size=rnd_B_size)
        self.rndBy = generator.normal(loc=0, scale=self.sigma, size=rnd_B_size)
        self.rndBz = generator.normal(loc=0, scale=self.sigma, size=rnd_B_size)
        self.Sol = sc.integrate.solve_ivp(self.func_S, self.t, self.S0, t_eval=self.t_eval, atol=1e-9, rtol=1e-9)
        self.S = self.Sol.y
        return self.S




if __name__ == '__main__':
    S0 = [1, 0, 0]

    t = [0, 200]  # t(時間)が0〜100まで動き、その時のfを求める。
    t_eval = np.linspace(*t, 2000001)
    mu_0 = 1.2
    omega = 20.2244

    ther_dt = 0.01


    alpha = 0.05
    gamma = 0.17

    kb = [1.38, -23]
    sta_M = [1.4, 0]  # 飽和磁化(T)で入れる
    T = [3.0, 2]
    V = [np.pi * 60 * 60 * 1, -27]
    ns = [1, -9]
    gamma_use = [1.76/(2 *np.pi) , 11]
    mu_0 = [1.256, -6]
    Bthe_n = kb[0] * T[0] * mu_0[0] / (sta_M[0] * V[0] * gamma_use[0] * ns[0])
    Bthe_o = kb[1] + T[1] + mu_0[1] - (sta_M[1] + V[1] + gamma_use[1] + ns[1])
    Bthe_2 = 2 * alpha * Bthe_n * (10 ** Bthe_o)
    sigma_Bthe = np.sqrt(Bthe_2) * 1000 * ther_dt# 最後の1000はmTにするため
    print(sigma_Bthe)

    spin = Thermal_spin(alpha, gamma, [0, 0, 0], S0, t, t_eval, [0, 1, 0], [0, 20.232, 0], [0, 0, 0], 1, 0, 0, 0, sigma_Bthe,ther_dt, 0,500)
    #spin.doit()
    spin.history()
    spin.poincore(1,900,1000)
    spin.tSi_graph(0,100,[1,2])
    spin.make_gif()
    spin.make_trajectory(100, 1000)
