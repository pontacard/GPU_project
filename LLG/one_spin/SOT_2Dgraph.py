import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from SOT import SOT
class SOT_2D(SOT):
    def __init__(self,alpha,gamma,B,S0,t,t_eval,spin_flow,Kx,Ky,Kz,beta,start,stop):
        super().__init__(alpha,gamma,B,S0,t,t_eval,spin_flow,Kx,Ky,Kz,beta,start,stop)


    def get_graph(self):
        self.Sol = sc.integrate.solve_ivp(self.func_S, self.t, self.S0, t_eval=self.t_eval)
        self.S = self.Sol.y
        t = self.Sol.t
        #print(t)

        x = self.S[0]
        y = self.S[1]
        z = self.S[2]
        S = np.sqrt(x*x + y*y + z*z)
        #print(S)
        #plt.plot(t, S, label = "|S|", ls = "--")
        #plt.savefig("reverse_|S|.png")
        #plt.plot(t,x)
        plt.plot(t, x, label="Sx")
        #plt.savefig("reverse_x.png")
        #plt.show()
        plt.plot(t, y, label = "Sy")
        #plt.savefig("reverse_y.png")
        #plt.show()
        plt.plot(t, z,  label = "Sz")
        plt.ylim(-1.1,1.1)

        plt.xlabel("time(ns)")
        plt.ylabel("S")
        #plt.axhline(color = 'k')
        plt.legend()

        plt.savefig("TypeZ_Fukami_xyzS.pdf")
        plt.show()


if __name__ == '__main__':
    S0 = [0, 0, 1]

    t = [0, 0.5]  # t(時間)が0〜100まで動き、その時のfを求める。
    t_eval = np.linspace(*t, 2000)

    plotB = [[0, 0, -1.2], [0, 0, 2.4]]

    mu_0 = 1.2
    gamma = 2.8
    h_div_2e = [0.329, -15]
    sta_M = [1.4, 0]  # 飽和磁化(T)で入れる
    theta = [-2.5, -1]
    j = [22, 11]
    d = [1.2, -9]
    Hsn = h_div_2e[0] * theta[0] * j[0] / (sta_M[0] * d[0])
    Hso = h_div_2e[1] + theta[1] + j[1] - (sta_M[1] + d[1])
    Hs = Hsn * (10 ** Hso) * 1000 * (mu_0 / 1200000) * mu_0  # 最後の1000はmTにするため
    print(Hs)

    spin = SOT_2D(0.02, gamma, [mu_0 * 2, 0, 0], S0, t, t_eval, [0, -Hs, 0], 0, 0, mu_0 * 250, 0, 0, 0.06)
    spin.get_graph()