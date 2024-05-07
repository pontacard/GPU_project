import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from chaos.tool import Tool
import random


class Thermal_FMR(Tool):
    def __init__(self, alpha, gamma,B,K, B0,omega,phase,sigma,ther_dt, t,  t_eval, S0):
        super().__init__(t,  t_eval)
        self.alpha = alpha
        self.B = B
        self.K = K
        self.gamma = gamma
        self.B0 = B0
        self.omega = omega
        self.phase = phase
        self.sigma = sigma
        self.ther_dt = ther_dt
        self.S0 = S0
        self.dt = []
        self.X0 = self.S0

    def func(self,t,S):
        self.t_check = t // self.ther_dt  # 時間が細かい実数値をとるから、幅ther_dtで時間を離散化する
        BT_num = int(t // self.ther_dt)
        # print(self.rndBx,t // self.ther_dt)

        self.BT = [self.rndBr[BT_num], self.rndBtheta[BT_num], self.rndBphi[BT_num]]
        sinth = np.sin(S[0])
        costh = np.cos(S[0])
        sinph = np.sin(S[1])
        cosph = np.cos(S[1])
        B_theta = costh * cosph * (self.B[0] + self.B0[0] * np.sin(S[2] + self.phase[0]) + self.K[0] * sinth * cosph) + costh *sinph * (self.B[1] + self.B0[1] * np.sin(S[2] + self.phase[1])+ self.K[1] * sinth * sinph) - sinth * (self.B[2] + self.B0[2] * np.sin(S[2] + self.phase[2]) + self.K[2] * costh) + self.BT[1]
        B_phi = - sinph * (self.B[0] + self.B0[0] * np.sin(S[2] + self.phase[0]) + self.K[0] * sinth * cosph) + cosph * (self.B[1] + self.B0[1] * np.sin(S[2] + self.phase[1]) + self.K[1] * sinth * sinph) + self.BT[2]
        dtdth = self.gamma * B_phi + self.alpha * self.gamma * B_theta
        dtdph = - (self.gamma * B_theta / sinth) + (self.alpha * self.gamma * B_phi / sinth)
        dtdo = self.omega
        dtdfunc = [dtdth,dtdph, dtdo]
        #print(dtdfunc)
        #print(t)
        return dtdfunc

    def history(self):
        rnd_B_size = int(self.t[1] // self.ther_dt) + 1
        generator = np.random.default_rng()
        self.rndBr = generator.normal(loc=0, scale=self.sigma, size=rnd_B_size)
        self.rndBtheta = generator.normal(loc=0, scale=self.sigma, size=rnd_B_size)
        self.rndBphi = generator.normal(loc=0, scale=self.sigma, size=rnd_B_size)
        self.Sol = sc.integrate.solve_ivp(self.func, self.t, self.X0, t_eval=self.t_eval, atol=1e-9, rtol=1e-9)
        return self.Sol

    def matsunaga_Lyapunov(self,pertu,step,cal_rate):
        rnd_B_size = int(self.t[1] // self.ther_dt) + 1
        generator = np.random.default_rng()
        self.rndBr = generator.normal(loc=0, scale=self.sigma, size=rnd_B_size)
        self.rndBtheta = generator.normal(loc=0, scale=self.sigma, size=rnd_B_size)
        self.rndBphi = generator.normal(loc=0, scale=self.sigma, size=rnd_B_size)

        dt = (self.t[1] - self.t[0]) / len(self.t_eval)
        Lya_dt = int(len(self.t_eval) / step)
        dX0 = np.array(self.X0) + np.array(pertu)
        Sol0 = sc.integrate.solve_ivp(self.func, self.t, self.X0, t_eval=self.t_eval, atol=1e-9, rtol=1e-9)
        t_evalp = np.linspace(*[0, 20 * dt * Lya_dt], Lya_dt * 20)
        Solp = sc.integrate.solve_ivp(self.func, [0, 20 * dt * Lya_dt], dX0, t_eval=t_evalp, atol=1e-9, rtol=1e-9)
        #print(Sol0.y)
        ans0 = Sol0.y.T
        ansp = Solp.y.T

        #print(ans0, ansp)
        dist0 = np.linalg.norm(ans0[0][:-1] - ansp[0][:-1])

        Lya = 0

        for i in range(step - 1):
            ansp[Lya_dt][-1] = ans0[(i + 1) * Lya_dt][-1]
            pi = np.linalg.norm(ans0[(i + 1) * Lya_dt] - ansp[Lya_dt]) / dist0
            #print(i,pi)
            #print(ans0[(i + 1) * Lya_dt],ansp[Lya_dt])
            per_X0i = ans0[(i + 1) * Lya_dt] + (ansp[Lya_dt] - ans0[(i + 1) * Lya_dt])/pi
            #per_X0i[-1] = (i + 1) * dt * Lya_dt * self.omega
            tp = [self.t[0],self.t[1] * cal_rate]
            #print(tp)
            eval_len = int((len(self.t_eval) - 1) * cal_rate + 1)
            #print(eval_len)
            t_evalp = np.linspace(*tp,eval_len)
            #print(tp,t_evalp)
            Solp = sc.integrate.solve_ivp(self.func, tp, per_X0i, t_eval = t_evalp, atol=1e-9, rtol=1e-9)
            ansp = Solp.y.T
            Lya += np.log(pi)
            #print(Lya)

        Lyap_expo = Lya/self.t[1]
        print(Lyap_expo)
        return Lyap_expo

if __name__ == '__main__':
    t = [0,100]
    t_eval = np.linspace(*t, 500001)#1の位は1にしないと時間ステップが有理数にならなくなる→誤差が大きくなり、正しい結果が得れない

    omega = 20.2244

    ther_dt = 0.01

    alpha = 0.05
    gamma = 0.17

    kb = [1.38, -23]
    sta_M = [1.4, 0]  # 飽和磁化(T)で入れる
    T = [3.0, 4]
    V = [np.pi * 60 * 60 * 1, -27]
    ns = [1, -9]
    gamma_use = [1.76 / (2 * np.pi), 11]
    mu_0 = [1.256, -6]
    Bthe_n = kb[0] * T[0] * mu_0[0] / (sta_M[0] * V[0] * gamma_use[0] * ns[0])
    Bthe_o = kb[1] + T[1] + mu_0[1] - (sta_M[1] + V[1] + gamma_use[1] + ns[1])
    Bthe_2 = 2 * alpha * Bthe_n * (10 ** Bthe_o)
    sigma_Bthe = np.sqrt(Bthe_2) * 1000 * ther_dt  # 最後の1000はmTにするため
    print(sigma_Bthe)


    B = [165,0,0]
    K = [0, 200, 0]
    Bac_Amp = [0, 9, 0]
    Bac_phase = [0, 0, 0]
    duff = Thermal_FMR(0.05,0.17,B,K,Bac_Amp,20.232,Bac_phase,sigma_Bthe,ther_dt,t,t_eval,[np.pi/2,0,0])
    duff.history()
    duff.diff_phase_graph(1,'φ','dφ/dt',490000,500000)
    #duff.Lyapunov([0.001,0.001,0],1000,35000)
    cal_rate = 0.01
    duff.FMR_matsunaga_Lyapunov([0.01,0,0],200,cal_rate)
