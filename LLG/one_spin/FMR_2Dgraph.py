import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from FMR import FMR_spin
class one_spin_2D(FMR_spin):
    def __init__(self,alpha,gamma,B,S0,t,t_eval,Amp,omega,theta,Kx,Ky,Kz,beta,start,stop):
        super().__init__(alpha,gamma,B,S0,t,t_eval,Amp,omega,theta,Kx,Ky,Kz,beta,start,stop)


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
        #plt.plot(t, y, label="Sy")
        plt.plot(t,z , label = 'Sz')
        plt.show()

        N = len(x[350000:])  # サンプル数
        f_s = 400  # サンプリングレート f_s[Hz] (任意)
        dt = (self.t[1] - self.t[0]) / len(self.t_eval)  # サンプリング周期 dt[s]

        y_fft = np.fft.fft(x[350000:])  # 離散フーリエ変換
        freq = np.fft.fftfreq(N, d=dt)  # 周波数を割り当てる（※後述）
        Amp = abs(y_fft / (N / 2))  # 音の大きさ（振幅の大きさ）
        plt.plot(freq[1:int(N / 2)], Amp[1:int(N / 2)])  # A-f グラフのプロット
        # plt.xscale("log")  # 横軸を対数軸にセット
        plt.xlim(0, 20)
        plt.xlabel("GHz")

        plt.show()
        plt.plot(t, y, label = "Sy")
        #plt.savefig("reverse_y.png")
        #plt.show()
        #plt.plot(t, z,  label = "Sz")
        plt.ylim(-1.1,1.1)

        plt.xlabel("time(ns)")
        plt.ylabel("S")
        #plt.axhline(color = 'k')
        plt.legend()

        #plt.savefig("no_dump_xy.pdf")
        plt.show()



if __name__ == '__main__':
    S0 = [3/5, 4/5, 0]

    t = [0, 300]  # t(時間)が0〜100まで動き、その時のfを求める。
    t_eval = np.linspace(*t, 600000)
    mu_0 = 1.2
    B0 = 10.5
    Bx = 160
    omega = 20.2244

    spin = one_spin_2D(0.05, 0.17, [Bx,0,0],S0,t,t_eval,[0,B0,0],[0,omega,0],[0,0,0],0 , 200, 0, 0, 0, 9000)
    spin.get_graph()
    spin.poincare(300000,600000)
