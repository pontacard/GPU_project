import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from one_spin.one_spin_LLG import A_spin

class STO_spin(A_spin):
    def __init__(self,alpha,gamma,B,S0,t, t_eval,H_Amp,omega,theta,Kx,Ky,Kz,beta,start,stop,STO_ac_effH,STO_dc_effH,lambdaa,eta):
        super().__init__(alpha, gamma, B, S0, t, t_eval)
        self.Amp = H_Amp
        self.omega = omega
        self.Kx = Kx
        self.Ky = Ky
        self.Kz = Kz
        self.beta = beta
        self.start = start
        self.stop = stop
        self.theta = theta
        self.STO_ac_effH = STO_ac_effH
        self.STO_dc_effH = STO_dc_effH
        self.lambdaa = lambdaa
        self.eta = eta
        self.STO_effH = 0



    def func_S(self,t,S):  # 関数を設定
        Snorm = np.linalg.norm(S)
        Bk = np.array([self.Kx * S[0] / (Snorm * Snorm), self.Ky * S[1] / (Snorm * Snorm),self.Kz * S[2] / (Snorm * Snorm)])
        sot_torque = [0,0,0]
        p = np.array([0,1,0])
        SOT_angle = 0
        #print(S,p,np.dot(S,p))
        if t >= self.start and t <= self.stop:
            self.STO_effH = [self.STO_dc_effH[0] + self.STO_ac_effH[0] * np.cos(self.omega[0] * t + self.theta[0]),
             self.STO_dc_effH[1] + self.STO_ac_effH[1] * np.cos(self.omega[1] * t + self.theta[1]),
             self.STO_dc_effH[2] + self.STO_ac_effH[2] * np.cos(self.omega[2] * t + self.theta[2])]

            B = np.array(self.B) + Bk + np.array([self.Amp[0] * np.cos(self.omega[0] * t +  self.theta[0]) , self.Amp[1] * np.cos(self.omega[1] * t + self.theta[1]),0])
            #print(B)
            SOT_angle = self.eta/(1 + self.lambdaa * np.dot(S,p))
            #print(SOT_angle * self.STO_effH[1])
            sot_torque = [self.gamma  * (S[1] * (self.STO_effH[0] * S[1] - self.STO_effH[1] * S[0]) - S[2] * (
                    self.STO_effH[2] * S[0] - self.STO_effH[0] * S[2])), self.gamma * (
                                 S[2] * (self.STO_effH[1] * S[2] - self.STO_effH[2] * S[1]) - S[0] * (
                                 self.STO_effH[0] * S[1] - self.STO_effH[1] * S[0])), self.gamma * (
                                 S[0] * (self.STO_effH[2] * S[0] - self.STO_effH[0] * S[2]) - S[1] * (
                                 self.STO_effH[1] * S[2] - self.STO_effH[2] * S[1]))]
        else:
            B = np.array(self.B) + Bk

        Snorm = np.linalg.norm(S)
        dSxdt = - self.gamma * (B[1] * S[2] - B[2] * S[1]) - SOT_angle * sot_torque[0] - (self.gamma * self.alpha/Snorm) * (S[1] * (S[0] * B[1] - S[1] * B[0]) - S[2]* (S[2] * B[0] - S[0] * B[2]))
        dSydt = - self.gamma * (B[2] * S[0] - B[0] * S[2]) - SOT_angle * sot_torque[1] - (self.gamma * self.alpha/Snorm) * (S[2] * (S[1] * B[2] - S[2] * B[1]) - S[0]* (S[0] * B[1] - S[1] * B[0]))
        dSzdt = - self.gamma * (B[0] * S[1] - B[1] * S[0]) - SOT_angle * sot_torque[2] - (self.gamma * self.alpha/Snorm) * (S[0] * (S[2] * B[0] - S[0] * B[2]) - S[1]* (S[1] * B[2] - S[2] * B[1]))
        dSdt = [dSxdt, dSydt, dSzdt]

        return dSdt

if __name__ == '__main__':
    S0 = [0, 0, 1]

    t = [0, 20]  # t(時間)が0〜100まで動き、その時のfを求める。
    t_eval = np.linspace(*t, 1000)

    plotB = [[0, 0, -1.2], [0, 0, 2.4]]

    gamma = 0.17
    mu_0 = 1.26
    mu_h_div_2e = [0.413563, -21]
    sta_M = [1.824, 0]  # 飽和磁化(T)で入れる
    jac = [0, 11]
    # jac = [0.05, 11]
    d = [2, -9]
    Hacn = mu_h_div_2e[0] * jac[0] / (sta_M[0] * d[0])
    Haco = mu_h_div_2e[1] + jac[1] - (sta_M[1] + d[1])
    Hac = Hacn * (10 ** Haco) * 1000 / gamma  # 最後の1000はmTにするため
    print(Hac)

    jdc = [0.5, 11]
    Hdcn = mu_h_div_2e[0] * jdc[0] / (sta_M[0] * d[0])
    Hdco = mu_h_div_2e[1] + jdc[1] - (sta_M[1] + d[1])
    Hdc = Hdcn * (10 ** Hdco) * 1000 / gamma  # 最後の1000はmTにするため

    Kx = 200
    Ky = 0
    Kz = 0

    spin = STO_spin(0.005, gamma, [0, 0, 160], S0, t, t_eval, [0, 0, 0], [0, 5.5, 0], [0, 0, 0], Kx, Ky,
                   mu_0 * Kz, 0, 0, 20, [0, Hac, 0], [0, Hdc, 0], 0.288, 0.537)
    print(spin.history())
    spin.doit()