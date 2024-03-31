
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from one_spin.one_spin_LLG import A_spin

class FMR_spin(A_spin):
    def __init__(self,alpha,gamma,B,S0,t, t_eval,Amp,omega,theta,Kx,Ky,Kz,beta,start,stop):
        super().__init__(alpha, gamma, B, S0, t, t_eval)
        self.Amp = Amp
        self.omega = omega
        self.Kx = Kx
        self.Ky = Ky
        self.Kz = Kz
        self.beta = beta
        self.start = start
        self.stop = stop
        self.theta = theta

    def func_S(self,t,S):  # 関数を設定
        Snorm = np.linalg.norm(S)
        Bk = np.array([self.Kx * S[0] / (Snorm * Snorm), self.Ky * S[1] / (Snorm * Snorm),self.Kz * S[2] / (Snorm * Snorm)])
        if t >= self.start and t <= self.stop:
            B = np.array(self.B) + Bk + np.array([self.Amp[0] * np.cos(self.omega[0] * t +  self.theta[0]) , self.Amp[1] * np.sin(self.omega[1] * t + self.theta[1]),self.Amp[2] * np.sin(self.omega[2] * t + self.theta[2])])
            #print(B)
        else:
            B = np.array(self.B) + Bk

        Snorm = np.linalg.norm(S)
        dSxdt = - self.gamma * (S[1] * B[2] - S[2] * B[1]) - (self.gamma * self.alpha/Snorm) * (S[1] * (S[0] * B[1] - S[1] * B[0]) - S[2]* (S[2] * B[0] - S[0] * B[2]))
        dSydt = - self.gamma * (S[2] * B[0] - S[0] * B[2]) - (self.gamma * self.alpha/Snorm) * (S[2] * (S[1] * B[2] - S[2] * B[1]) - S[0]* (S[0] * B[1] - S[1] * B[0]))
        dSzdt = - self.gamma * (S[0] * B[1] - S[1] * B[0]) - (self.gamma * self.alpha/Snorm) * (S[0] * (S[2] * B[0] - S[0] * B[2]) - S[1]* (S[1] * B[2] - S[2] * B[1]))
        dSdt = [dSxdt, dSydt, dSzdt]

        return dSdt

    def poincore(self,start,stop):
        his = self.history()
        x = self.Sol.y[0]
        t = self.t_eval
        dt = (self.t[1] - self.t[0]) / len(self.t_eval)  # サンプリング周期 dt[s]
        T = 2 * np.pi / self.omega[1]
        storobo = int(T / dt)
        x_storobo = x[start:stop:storobo]

        return x_storobo


if __name__ == '__main__':
    S0 = [4/5, 0, 3/5]

    t = [0,15] # t(時間)が0〜100まで動き、その時のfを求める。
    t_eval = np.linspace(*t, 1000)
    mu_0 = 1.2

    spin = FMR_spin(0.01, 0.17, [0,0,0],S0,t,t_eval,[0,0,10000],[0,0,6.8],[0,0,0],0 ,0 ,mu_0 * 10, 0, 0, 15)
    spin.doit()
