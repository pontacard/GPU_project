import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from macro_spin.tool import Tool


class FMR_spin(Tool):
    def __init__(self, alpha, gamma, B, S0, t, t_eval, Amp, omega, theta, Kx, Ky, Kz, beta, start, stop):
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

    def func_S(self, t, S):  # 関数を設定
        Snorm = np.linalg.norm(S)
        Bk = np.array(
            [self.Kx * S[0] / (Snorm * Snorm), self.Ky * S[1] / (Snorm * Snorm), self.Kz * S[2] / (Snorm * Snorm)])
        if t >= self.start and t <= self.stop:
            B = np.array(self.B) + Bk + np.array([self.Amp[0] * np.cos(self.omega[0] * t + self.theta[0]),
                                                  self.Amp[1] * np.sin(self.omega[1] * t + self.theta[1]),
                                                  self.Amp[2] * np.sin(self.omega[2] * t + self.theta[2])])
            # print(B)
        else:
            B = np.array(self.B) + Bk

        Snorm = np.linalg.norm(S)
        dSxdt = - self.gamma * (S[1] * B[2] - S[2] * B[1]) - (self.gamma * self.alpha / Snorm) * (
                    S[1] * (S[0] * B[1] - S[1] * B[0]) - S[2] * (S[2] * B[0] - S[0] * B[2]))
        dSydt = - self.gamma * (S[2] * B[0] - S[0] * B[2]) - (self.gamma * self.alpha / Snorm) * (
                    S[2] * (S[1] * B[2] - S[2] * B[1]) - S[0] * (S[0] * B[1] - S[1] * B[0]))
        dSzdt = - self.gamma * (S[0] * B[1] - S[1] * B[0]) - (self.gamma * self.alpha / Snorm) * (
                    S[0] * (S[2] * B[0] - S[0] * B[2]) - S[1] * (S[1] * B[2] - S[2] * B[1]))
        dSdt = [dSxdt, dSydt, dSzdt]

        return dSdt




if __name__ == '__main__':
    S0 = [1, 0, 0]

    t = [0, 10]  # t(時間)が0〜100まで動き、その時のfを求める。
    t_eval = np.linspace(*t, 1000)
    mu_0 = 1.2
    omega = 20.2244

    spin = FMR_spin(0.05, 0.17, [165, 0, 0], S0, t, t_eval, [0, 11.4, 0], [0, 20.232, 0], [0, 0, 0], 0, 200, 0, 0, 0,15)
    spin.doit()
    spin.history()
    spin.poincore(1,900,1000)
    # spin.tSi_graph(0,8,[1,2])
    spin.make_gif()
    spin.make_trajectory(100, 1000)
