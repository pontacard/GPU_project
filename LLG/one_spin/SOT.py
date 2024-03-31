import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from one_spin_LLG import A_spin

class SOT(A_spin):
    def __init__(self,alpha,gamma,B,S0,t,t_eval,spin_flow,Kx,Ky,Kz,beta,start,stop): #scはSlonczewski-like torque、fiはfield-like torqueをそれぞれ実行的磁場と考えた時のバイアス
        super().__init__(alpha,gamma,B,S0,t,t_eval)
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
        if t >= self.start and t <= self.stop:
            B_e = [self.B[0] + self.beta * self.spin_flow[0] + self.Kx * S[0] / (Snorm * Snorm),
                   self.B[1] + self.beta * self.spin_flow[1] + self.Ky * S[1] / (Snorm * Snorm),
                   self.B[2] + self.beta * self.spin_flow[2] + self.Kz * S[2] / (Snorm * Snorm)]
            B_a = [self.B[0] + self.beta * self.spin_flow[0], self.B[1] + self.beta * self.spin_flow[1],
                   self.B[2] + self.beta * self.spin_flow[2]]
            sc_torque = [self.gamma * (S[1] * (self.spin_flow[0] * S[1] - self.spin_flow[1] * S[0]) - S[2] * (
                    self.spin_flow[2] * S[0] - self.spin_flow[0] * S[2])), self.gamma * (
                                 S[2] * (self.spin_flow[1] * S[2] - self.spin_flow[2] * S[1]) - S[0] * (
                                 self.spin_flow[0] * S[1] - self.spin_flow[1] * S[0])), self.gamma * (
                                 S[0] * (self.spin_flow[2] * S[0] - self.spin_flow[0] * S[2]) - S[1] * (
                                 self.spin_flow[1] * S[2] - self.spin_flow[2] * S[1]))]
        else:
            B_e = [self.B[0] + self.Kx * S[0] / (Snorm * Snorm), self.B[1] + self.Ky * S[1] / (Snorm * Snorm),
                   self.B[2] + self.Kz * S[2] / (Snorm * Snorm)]
            B_a = [self.B[0], self.B[1], self.B[2]]
            sc_torque = [0, 0, 0]

        Snorm = np.linalg.norm(S)

        dSxdt = - self.gamma * (B_e[2] * S[1] - B_e[1] * S[2]) - sc_torque[0] - (self.alpha / Snorm) * (
                S[1] * (self.gamma * (B_e[1] * S[0] - B_e[0] * S[1])) - S[2] * (
                self.gamma * (B_e[2] * S[0] - B_e[0] * S[2])))
        dSydt = - self.gamma * (B_e[0] * S[2] - B_e[2] * S[0]) - sc_torque[1] - (self.alpha / Snorm) * (
                S[2] * (self.gamma * (B_e[2] * S[1] - B_e[1] * S[2])) - S[0] * (
                self.gamma * (B_e[0] * S[1] - B_e[1] * S[0])))
        dSzdt = - self.gamma * (B_e[1] * S[0] - B_e[0] * S[1]) - sc_torque[2] - (self.alpha / Snorm) * (
                S[0] * (self.gamma * (B_e[0] * S[2] - B_e[2] * S[0])) - S[1] * (
                self.gamma * (B_e[1] * S[2] - B_e[2] * S[1])))
        dSdt = [dSxdt, dSydt, dSzdt]
        # print(dSdt)

        return dSdt


if __name__ == '__main__':
    S0 = [1, 0, 0]

    t = [0, 4]  # t(時間)が0〜100まで動き、その時のfを求める。
    t_eval = np.linspace(*t, 1200)

    plotB = [[0, 0, -1.2], [0, 0, 2.4]]

    gamma = 0.22
    mu_h_div_2e = [0.824, -21]
    sta_M = [1.4, 0]  # 飽和磁化(T)で入れる
    theta = [-2, -1]
    j = [2.5, 11]
    d = [1, -9]
    Hsn = mu_h_div_2e[0] * theta[0] * j[0] / (sta_M[0] * d[0])
    Hso = mu_h_div_2e[1] + theta[1] + j[1] - (sta_M[1] + d[1])
    Hs = Hsn * (10 ** Hso) * 1000 / gamma  # 最後の1000はmTにするため
    print(Hs)

    spin = SOT(0.02, gamma, [0, 0, -55], S0, t, t_eval, [0, Hs, 0], 20, 0, -500, 0, 0, 0.5)
    spin.doit()