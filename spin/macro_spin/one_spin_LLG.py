import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from macro_spin.tool import Tool

class A_spin(Tool):
    def __init__(self,alpha,gamma,B,S0,t, t_eval):
        super().__init__(t,t_eval)
        self.alpha = alpha  #緩和項をどれくらい大きく入れるか
        self.gamma = gamma  #LLg方程式のγ
        self.B = B          #外部磁場(tで時間変化させてもいいよ)
        self.S0 = S0        #初期のスピンの向き
        self.t = t          #どれくらいの時間でやるか
        self.t_eval = t_eval #ステップ数はどうするか
        self.S = []
        self.ax = 0         #pltのオブジェクト
        self.fig = 0        #pltのオブジェクト
        self.quiveraa = 0

    def func_S(self,t,S):  # 関数を設定
        #print(S)
        B = self.B
        Snorm = np.linalg.norm(S)
        dSxdt = - self.gamma * (S[1] * B[2] - S[2] * B[1]) - (self.gamma * self.alpha/Snorm) * (S[1] * (S[0] * B[1] - S[1] * B[0]) - S[2]* (S[2] * B[0] - S[0] * B[2]))
        dSydt = - self.gamma * (S[2] * B[0] - S[0] * B[2]) - (self.gamma * self.alpha/Snorm) * (S[2] * (S[1] * B[2] - S[2] * B[1]) - S[0]* (S[0] * B[1] - S[1] * B[0]))
        dSzdt = - self.gamma * (S[0] * B[1] - S[1] * B[0]) - (self.gamma * self.alpha/Snorm) * (S[0] * (S[2] * B[0] - S[0] * B[2]) - S[1]* (S[1] * B[2] - S[2] * B[1]))
        dSdt = [dSxdt, dSydt, dSzdt]

        return dSdt

if __name__ == '__main__':
    S0 = [0.8, 0.6, 0]

    t = [0,150] # t(時間)が0〜100まで動き、その時のfを求める。
    t_eval = np.linspace(*t, 1000)

    spin = A_spin(0.01,-0.1,[0,0,-5],S0,t,t_eval)
    spin.doit()
    spin.history()
    spin.tSi_graph(0,80,[0,2])
    spin.Si_fourier(0,80,1)
