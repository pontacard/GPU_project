import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from chaos.tool import Tool


class FMR_Duffing(Tool):
    def __init__(self, alpha, gamma,Bx,Ky, B0,omega, t,  t_eval, S0):
        super().__init__(t,  t_eval)
        self.alpha = alpha
        self.Bx = Bx
        self.Ky = Ky
        self.gamma = gamma
        self.B0 = B0
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
        dtdth = self.gamma * (-self.Bx * sinph + self.Ky  * sinth * sinph * cosph  + self.B0  * cosph * np.sin(S[2])) + self.alpha * self.gamma * (self.Bx * costh * cosph  + self.Ky * sinth * costh * sinph * sinph + self.B0  * sinph * np.sin(S[2]) * costh)
        dtdph = self.gamma * (-self.Bx * costh * cosph / (sinth) - self.Ky * costh * sinph * sinph - self.B0  * sinph * np.sin(S[2]) * costh/ sinth) + self.alpha * self.gamma * (-self.Bx * sinph / sinth + self.Ky  * sinph * cosph  + self.B0  * cosph * np.sin(S[2])/sinth)
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
    t = [0,10]
    t_eval = np.linspace(*t, 1000)#1の位は1にしないと時間ステップが有理数にならなくなる→誤差が大きくなり、正しい結果が得れない
    duff = FMR_Duffing(0.05,0.17,165,200,11.4,20.232,t,t_eval,[np.pi/2,0,0],0.1)
    duff.make_Ani()
    #duff.diff_phase_graph(1,'φ','dφ/dt',190000,200000)
    #duff.Lyapunov([0.001,0.001,0],1000,35000)
    cal_rate = 0.01
    duff.FMR_matsunaga_Lyapunov([0.01,0,0],200,cal_rate)
