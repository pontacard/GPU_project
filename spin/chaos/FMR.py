import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from chaos.tool import Tool


class FMR(Tool):
    def __init__(self, alpha, gamma,B,K, B0,omega,phase, t,  t_eval, S0):
        super().__init__(t,  t_eval)
        self.alpha = alpha
        self.B = B
        self.K = K
        self.gamma = gamma
        self.B0 = B0
        self.omega = omega
        self.phase = phase
        self.S0 = S0
        self.dt = []
        self.X0 = self.S0

    def func(self,t,S):
        #print(S)
        sinth = np.sin(S[0])
        costh = np.cos(S[0])
        sinph = np.sin(S[1])
        cosph = np.cos(S[1])
        B_theta = costh * cosph * (self.B[0] + self.B0[0] * np.sin(S[2] + self.phase[0]) + self.K[0] * sinth * cosph) + costh *sinph * (self.B[1] + self.B0[1] * np.sin(S[2] + self.phase[1])+ self.K[1] * sinth * sinph) - sinth * (self.B[2] + self.B0[2] * np.sin(S[2] + self.phase[2]) + self.K[2] * costh)
        B_phi = - sinph * (self.B[0] + self.B0[0] * np.sin(S[2] + self.phase[0]) + self.K[0] * sinth * cosph) + cosph * (self.B[1] + self.B0[1] * np.sin(S[2] + self.phase[1]) + self.K[1] * sinth * sinph)
        dtdth = self.gamma * B_phi + self.alpha * self.gamma * B_theta
        dtdph = - (self.gamma * B_theta / sinth) + (self.alpha * self.gamma * B_phi / sinth)
        dtdo = self.omega
        dtdfunc = [dtdth,dtdph, dtdo]
        #print(dtdfunc)
        #print(t)
        return dtdfunc

    def history(self):
        self.Sol = sc.integrate.solve_ivp(self.func, self.t, self.S0, t_eval=self.t_eval,atol=1e-12,rtol=1e-12)
        ans = self.Sol.y
        t = self.Sol.t
        dtdth = np.diff(ans[0], 1)
        dtdph = np.diff(ans[1], 1)
        return [ans[0], dtdth, ans[1], dtdph, t]



if __name__ == '__main__':
    t = [0,500]
    t_eval = np.linspace(*t, 500001)#1の位は1にしないと時間ステップが有理数にならなくなる→誤差が大きくなり、正しい結果が得れない
    B = [165,0,0]
    K = [0, 200, -1000]
    Bac_Amp = [0, 19, 0]
    Bac_phase = [0, 0, 0]
    duff = FMR(0.05,0.17,B,K,Bac_Amp,20.232,Bac_phase,t,t_eval,[np.pi/2,0,0])
    duff.make_Ani()
    duff.diff_phase_graph(1,'φ','dφ/dt',490000,500000)
    #duff.Lyapunov([0.001,0.001,0],1000,35000)
    cal_rate = 0.01
    duff.FMR_matsunaga_Lyapunov([0.01,0,0],200,cal_rate)
