import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from chaos.tool import Tool


class SOT_Duffing(Tool):
    def __init__(self, alpha, gamma,Bx,Ky, SOT,omega, t,  t_eval, S0):
        super().__init__(t, t_eval)
        self.alpha = alpha
        self.Bx = Bx
        self.Ky = Ky
        self.gamma = gamma
        self.SOT = SOT #磁場換算した場合のSOTの振幅(gamma * SOTでトルクとなる)
        self.omega = omega
        self.S0 = S0
        self.dt = []
        self.X0 = self.S0

    def func(self,t,S):
        #print(S)
        sinth = np.sin(S[0])
        costh = np.cos(S[0])
        sinph = np.sin(S[1])
        cosph = np.cos(S[1])
        dtdth = self.gamma * (-self.Bx * sinph + self.Ky  * sinth * sinph * cosph  + self.SOT * np.sin(S[2]) * costh * sinph) + self.alpha * self.gamma * (self.Bx * costh * cosph  + self.Ky * sinth * costh * sinph * sinph)
        dtdph = self.gamma * (-self.Bx * costh * cosph / (sinth) - self.Ky * costh * sinph * sinph + self.SOT  *  np.sin(S[2]) * cosph/ sinth) + self.alpha * self.gamma * (-self.Bx * sinph / sinth + self.Ky * sinph * cosph)
        dtdo = self.omega
        dtdfunc = [dtdth,dtdph, dtdo]
        #print(dtdfunc)
        #print(t)
        return dtdfunc


if __name__ == '__main__':
    t = [0,500]
    t_eval = np.linspace(*t, 200001)
    duff = SOT_Duffing(0.005,0.17,165,200,27,20.232,t,t_eval,[np.pi/2,0.6435,0])
    duff.history()
    #duff.make_Ani()
    duff.diff_phase_graph(1,'φ','dφ/dt',190000,200000)
    cal_rate = 0.01
    duff.FMR_matsunaga_Lyapunov([0.01, 0, 0], 200, cal_rate)
