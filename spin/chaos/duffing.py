import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from chaos.tool import Tool

class Duffing(Tool):
    def __init__(self, alpha, beta, gamma, Amp,omega, t,  t_eval, X0):
        super().__init__(t, t_eval)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.Amp = Amp
        self.omega = omega
        self.X0 = X0

    def func(self,t,X):
        dtdx = X[1]
        dtdy =- self.alpha  * X[1] + self.beta * X[0] - self.gamma * (X[0] ** 3) + self.Amp * np.cos(X[2])
        dtdz = self.omega
        dtdfunc = [dtdx,dtdy,dtdz]
        return dtdfunc





if __name__ == '__main__':
    t = [0,200]
    t_eval = np.linspace(*t, 600000)
    duff = Duffing(1,32,132,7,3.5,t,t_eval,[0.8903,0,0])
    duff.history()
    duff.phase_graph(0,1,"x","y",400000,500000)
    #duff.Lyapunov([0,0,0.01],1000)
    #duff.poincore(300000,600000)
    duff.Lyapunov([0.01,0,0],200)
    duff.matsunaga_Lyapunov([0.01,0,0],200)
