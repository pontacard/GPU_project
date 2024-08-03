import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from macro_spin.one_spin_LLG import Tool

class Sin_SOT(Tool):
    def __init__(self,alpha,gamma,B,S0,t,t_eval,Amp,omega,theta,Kx,Ky,Kz,beta,start,stop): #scはSlonczewski-like torque、fiはfield-like torqueをそれぞれ実行的磁場と考えた時のバイアス
        super().__init__(t,t_eval)
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
        self.spin_flow = []


    def func_S(self,t,S):  # 関数を設定
        sc_torque = []
        self.spin_flow = [self.Amp[0] * np.sin(self.omega[0] * t + self.theta[0]), self.Amp[1] * np.sin(self.omega[1] * t + self.theta[1]), self.Amp[2] * np.sin(self.omega[2] * t + self.theta[2])]
        Snorm = np.linalg.norm(S)
        if t >= self.start and t <= self.stop:
            B_e = [self.B[0] + self.beta * self.spin_flow[0] + self.Kx * S[0] / (Snorm * Snorm),
                   self.B[1] + self.beta * self.spin_flow[1] + self.Ky * S[1] / (Snorm * Snorm),
                   self.B[2] + self.beta * self.spin_flow[2] + self.Kz * S[2] / (Snorm * Snorm)]
            B_a = [self.B[0] + self.beta * self.spin_flow[0], self.B[1] + self.beta * self.spin_flow[1],
                   self.B[2] + self.beta * self.spin_flow[2]]
            sc_torque = [self.gamma * (S[1] * (S[0] * self.spin_flow[1] - S[1] * self.spin_flow[0]) - S[2] * (
                    S[2] * self.spin_flow[0]  - S[0] * self.spin_flow[2])), self.gamma * (
                                 S[2] * (S[1] * self.spin_flow[2] - S[2] * self.spin_flow[1]) - S[0] * (
                                 S[0] * self.spin_flow[1] - S[1] * self.spin_flow[0])), self.gamma * (
                                 S[0] * (S[2] * self.spin_flow[0] - S[0] * self.spin_flow[2]) - S[1] * (
                                 S[1] * self.spin_flow[2] - S[2] * self.spin_flow[1]))]
        else:
            B_e = [self.B[0] + self.Kx * S[0] / (Snorm * Snorm), self.B[1] + self.Ky * S[1] / (Snorm * Snorm),
                   self.B[2] + self.Kz * S[2] / (Snorm * Snorm)]
            B_a = [self.B[0], self.B[1], self.B[2]]
            sc_torque = [0, 0, 0]

        Snorm = np.linalg.norm(S)

        dSxdt = - self.gamma * (S[1] * B_e[2] - S[2] * B_e[1]) - sc_torque[0] - (self.alpha / Snorm) * (
                S[1] * (self.gamma * (S[0] * B_e[1] -  S[1] * B_e[0])) - S[2] * (
                self.gamma * (S[2] * B_e[0]  -  S[0] * B_e[2])))
        dSydt = - self.gamma * (S[2] * B_e[0] - S[0] * B_e[2]) - sc_torque[1] - (self.alpha / Snorm) * (
                S[2] * (self.gamma * (S[1] * B_e[2] - S[2] * B_e[1])) - S[0] * (
                self.gamma * (S[0] * B_e[1] -  S[1] * B_e[0])))
        dSzdt = - self.gamma * (S[0] * B_e[1] -  S[1] * B_e[0]) - sc_torque[2] - (self.alpha / Snorm) * (
                S[0] * (self.gamma * (S[2] * B_e[0]  -  S[0] * B_e[2])) - S[1] * (
                self.gamma * (S[1] * B_e[2] - S[2] * B_e[1])))
        dSdt = [dSxdt, dSydt, dSzdt]
        # print(dSdt)

        return dSdt

if __name__ == '__main__':
    S0 = [4 / 5, 3 / 5, 0]

    t = [0, 800]  # t(時間)が0〜100まで動き、その時のfを求める。
    t_eval = np.linspace(*t, 8000001)

    # plotB = [[0, 0, -1.2], [0, 0, 2.4]]

    gamma = 0.176335977
    alpha = 0.05
    B = [160, 0, 0]
    SOT_Amp = [0, 16, 0]
    K = [0, 200, 0]

    spin1 = Sin_SOT(alpha, gamma, B, S0, t, t_eval, SOT_Amp, [0, 20.232, 0], [0, 0, 0], K[0], K[1], K[2], 0, 0, 9000)
    spin1.history()
    # spin1.tSi_graph(720,800,[1])
    # spin1.make_gif()
    savename = f'/Users/tatsumiryou/Spin_picture/for_paper/Magnetic_Duffing_Oscillator/raw_result/Bk={K[1]}_alt_mag_0.02/my_fourier_{B[0]}_alpha{alpha}_B0={SOT_Amp[1]}.pdf'
    spin1.Si_fourier(7000000, 8000000, 1, [0, 25], savefig=False, save_name=savename)
    savename = f'/Users/tatsumiryou/Spin_picture/for_paper/Magnetic_Duffing_Oscillator/raw_result/Bk={K[1]}_alt_spin/trajectry_{B[0]}_alpha{alpha}_B0={SOT_Amp[1]}.png'
    spin1.make_trajectory(7000000, 8000000, ticks=[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], savefig=False,
                          save_name=savename)
