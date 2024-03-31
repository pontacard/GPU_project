import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation

class Spin_wave():
    def __init__(self,alpha,gamma,B,S0,t,t_eval,K,sc,fi,pulse,N,J):
        self.alpha = alpha  # 緩和項をどれくらい大きく入れるか
        self.gamma = gamma  # LLg方程式のγ
        self.B = B  # 外部磁場(tで時間変化させてもいいよ)
        self.S0 = S0  # 初期のスピンの向き
        self.t = t  # どれくらいの時間でやるか
        self.t_eval = t_eval #ステップ数
        self.K = K #異方性磁気定数
        self.sc = sc #Slonczewski-like torqueの強さ
        self.fi = fi #field-like torqueの強さ
        self.pulse = pulse #[[開始時刻, 終了時刻, パルスの強さ], [開始時刻, 終了時刻, パルスの強さ],,,,]のようなパルス電流の情報
        self.J = J #交換相互作用定数
        self.N = N # スピンの数
        self.S = []
        self.ax = 0  # pltのオブジェクト
        self.fig = 0  # pltのオブジェクト
        self.quiveraa = 0
        self.quiver_dic = dict()

    def func_S(self,t,S):
        dSdt = np.empty(0)
        ba, fo = 0, 0
        B_sc = [self.B[0], self.B[1], self.B[2]]
        B_fi = [self.B[0], self.B[1], self.B[2]]


        for i in range(self.N):
            #print(S)
            Si = [S[3 * i + 0], S[3 * i + 1], S[3 * i + 2]]
            Sib = []
            Sif = []
            if i == 0:
                #print(t)
                for Bi in self.pulse:
                    fo = i + 1
                    Sib = [0, 0, 0]
                    Sif = [self.J * S[3 * fo + 0], self.J * S[3 * fo + 1], self.J * S[3 * fo + 2]]

                    if t > Bi[0] and t <= Bi[1]:
                        B_sc = [Sib[0] + Sif[0] + self.B[0] + self.sc * Bi[2][0], Sib[1] + Sif[1] + self.B[1] + self.sc * Bi[2][1],
                                Sib[2] + Sif[2] + self.B[2] + self.sc * Bi[2][2]]
                        B_fi = [Sib[0] + Sif[0] + self.B[0] + self.fi * Bi[2][0], Sib[1] + Sif[1] + self.B[1] + self.fi * Bi[2][1],
                                Sib[2] + Sif[2] + self.B[2] + self.fi * Bi[2][2]]
                        break
                    else:
                        B_sc = [Sib[0] + Sif[0] + self.B[0], Sib[1] + Sif[1] + self.B[1],Sib[2] + Sif[2] + self.B[2]]
                        B_fi = [Sib[0] + Sif[0] + self.B[0], Sib[1] + Sif[1] + self.B[1],Sib[2] + Sif[2] + self.B[2]]



                Snorm = np.linalg.norm(Si)
                KneZ = [self.K * Si[1] * Si[2] / (Snorm * Snorm), -self.K * Si[2] * Si[0] / (Snorm * Snorm), 0]

                dSixdt = - self.gamma * (B_sc[1] * Si[2] - B_sc[2] * S[1]) + KneZ[0] - (self.alpha / Snorm) * (
                            Si[1] * (self.gamma * (B_fi[0] * Si[1] - B_fi[1] * Si[0]) + KneZ[2]) - Si[2] * (
                                self.gamma * (B_fi[2] * Si[0] - B_fi[0] * Si[2]) + KneZ[1]))
                dSiydt = - self.gamma * (B_sc[2] * Si[0] - B_sc[0] * Si[2]) + KneZ[1] - (self.alpha / Snorm) * (
                            Si[2] * (self.gamma * (B_fi[1] * Si[2] - B_fi[2] * Si[1]) + KneZ[0]) - Si[0] * (
                                self.gamma * (B_fi[0] * Si[1] - B_fi[1] * Si[0]) + KneZ[2]))
                dSizdt = - self.gamma * (B_sc[0] * S[1] - B_sc[1] * S[0]) + KneZ[2] - (self.alpha / Snorm) * (
                            Si[0] * (self.gamma * (B_fi[2] * Si[0] - B_fi[0] * Si[2]) + KneZ[1]) - S[1] * (
                                self.gamma * (B_fi[1] * Si[2] - B_fi[2] * Si[1]) + KneZ[0]))
                #print(i, t, [dSixdt, dSiydt, dSizdt])


                dSdt = np.append(dSdt, [dSixdt, dSiydt, dSizdt])

            elif i == self.N - 1:
                ba = i - 1
                Sib = [self.J * S[3 * ba + 0], self.J * S[3 * ba + 1], self.J * S[3 * ba + 2]]
                Sif = [0, 0, 0]
                B = self.B

                Snorm = np.linalg.norm(S)
                KneZ = [self.K * Si[1] * Si[2] / (Snorm * Snorm), -self.K * Si[2] * Si[0] / (Snorm * Snorm), 0]

                ScrSSB = [(Si[1] * (Sib[2] + Sif[2]  -self.gamma * B[2]) - Si[2] * (Sib[1] + Sif[1] -self.gamma * B[1])),
                          (Si[2] * (Sib[0] + Sif[0] -self.gamma * B[0]) - Si[0] * (Sib[2] + Sif[2] -self.gamma * B[2])),
                          (Si[0] * (Sib[1] + Sif[1] -self.gamma * B[1]) - Si[1] * (Sib[0] + Sif[0] -self.gamma * B[0]))]



                dSixdt = ScrSSB[0] + KneZ[0] - (self.alpha / Snorm) * (
                        Si[1] * (ScrSSB[2] + KneZ[2]) - Si[2] * (ScrSSB[1] + KneZ[1]))
                dSiydt = ScrSSB[1] + KneZ[1] - (self.alpha / Snorm) * (
                        Si[2] * (ScrSSB[0] + KneZ[0]) - Si[0] * (ScrSSB[2] + KneZ[2]))
                dSizdt = ScrSSB[2] + KneZ[2] - (self.alpha / Snorm) * (
                        Si[0] * (ScrSSB[1] + KneZ[1]) - Si[1] * (ScrSSB[0] + KneZ[0]))

                dSdt = np.append(dSdt, [dSixdt, dSiydt, dSizdt])


            else:
                fo = i + 1
                ba = i - 1
                Sib = [S[3 * ba + 0], S[3 * ba + 1], S[3 * ba + 2]]
                Sif = [S[3 * fo + 0], S[3 * fo + 1], S[3 * fo + 2]]
                B = self.B

                Snorm = np.linalg.norm(S)
                KneZ = [self.K * Si[1] * Si[2] / (Snorm * Snorm), -self.K * Si[2] * Si[0] / (Snorm * Snorm), 0]

                ScrSSB = [
                    (Si[1] * (Sib[2] + Sif[2] - self.gamma * B[2]) - Si[2] * (Sib[1] + Sif[1] - self.gamma * B[1])),
                    (Si[2] * (Sib[0] + Sif[0] - self.gamma * B[0]) - Si[0] * (Sib[2] + Sif[2] - self.gamma * B[2])),
                    (Si[0] * (Sib[1] + Sif[1] - self.gamma * B[1]) - Si[1] * (Sib[0] + Sif[0] - self.gamma * B[0]))]

                dSixdt = ScrSSB[0] + KneZ[0] - (self.alpha / Snorm) * (
                        Si[1] * (ScrSSB[2] + KneZ[2]) - Si[2] * (ScrSSB[1] + KneZ[1]))
                dSiydt = ScrSSB[1] + KneZ[1] - (self.alpha / Snorm) * (
                        Si[2] * (ScrSSB[0] + KneZ[0]) - Si[0] * (ScrSSB[2] + KneZ[2]))
                dSizdt = ScrSSB[2] + KneZ[2] - (self.alpha / Snorm) * (
                        Si[0] * (ScrSSB[1] + KneZ[1]) - Si[1] * (ScrSSB[0] + KneZ[0]))

                dSdt = np.append(dSdt, [dSixdt, dSiydt, dSizdt])
                #print(i, t, [dSixdt, dSiydt, dSizdt])
        #print("fat",t,dSdt)
        #dSdt = np.reshape(dSdt, (1, -1))
        #print(dSdt)

        return dSdt


    # S0 = np.array([[1,0,0],[0,0,1],[0,0,1]])



    def get_spin_vec(self,t, i):
        Ox, Oy, Oz = 3 * i, 0, 0
        x = self.S[i*3][t]
        y = self.S[i*3 + 1][t]
        z = self.S[i*3 + 2][t]
        return Ox, Oy, Oz, x, y, z





    def update(self,t):
        for i in range(self.N):
            self.quiver_dic[i].remove()
            self.quiver_dic[i] = self.ax.quiver(*self.get_spin_vec(t, i))

    def doit(self):
        self.S0 = np.reshape(self.S0, (1, -1))
        self.S0 = list(self.S0[0])
        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection="3d"))

        self.Sol = sc.integrate.solve_ivp(self.func_S, self.t, self.S0, t_eval=self.t_eval)
        self.S = self.Sol.y
        #print(self.S)
        retS = np.reshape(self.S, (self.N,3,-1))
        frames = []

        #self.S = np.reshape(self.S, (-1, self.N, 3))

        for i in range(self.N):
            self.quiver_dic[i] = self.ax.quiver(*self.get_spin_vec(0, i))

        self.ax.set_xlim(-0, 99)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_zlim(-1.2, 1.2)
        #self.ax.view_init(elev=90)
        self.ax.view_init(azim= 240)

        ani = FuncAnimation(self.fig, self.update, frames=len(self.Sol.t), interval=1)
        ani.save("Spin_wave.gif",writer='imagemagick')
        plt.show()

        return retS

if __name__ == '__main__':
    n = 30
    S0 = np.zeros((n, 3))

    for i in range(n):
        S0[i][2] = 1
    #print(S0)

    t = [0, 50]  # t(時間)が0〜100まで動き、その時のfを求める。
    t_eval = np.linspace(*t, 100)

    spin = Spin_wave(0.001, 0.17, [0, 0, 0], S0, t, t_eval, 0.1, 0.1, 100, [[0,4,[0, -30, 0]], [200, 220,[0, -30, 0]]], n, 10)
    spin.doit()
