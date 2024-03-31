import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from chaos.New_Spin_Lyapunov import Lyapunov
from one_spin.FMR import FMR_spin
from one_spin.FMR_gif import FMR_gif

class Duffing():
    def __init__(self, alpha, beta, gamma, Amp,omega, t,  t_eval, X0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.Amp = Amp
        self.omega = omega
        self.t = t
        self.X0 = X0
        self.t_eval = t_eval

    def func(self,t,X):
        dtdx = X[1]
        dtdy =- self.alpha  * X[1] + self.beta * X[0] - self.gamma * (X[0] ** 3) + self.Amp * np.cos(X[2])
        dtdz = self.omega
        dtdfunc = [dtdx,dtdy,dtdz]
        return dtdfunc

    def history(self):
        self.Sol = sc.integrate.solve_ivp(self.func, self.t, self.X0, t_eval=self.t_eval, atol=1e-12, rtol=1e-12)
        return self.Sol

    def doit(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        self.Sol = sc.integrate.solve_ivp(self.func,self.t, self.X0, t_eval=self.t_eval,atol=1e-12,rtol=1e-12)
        ans = self.Sol.y

        print(ans)
        plt.plot(ans[0][450000:],ans[1][450000:])
        plt.xlabel("x")
        plt.ylabel("y")
        #plt.savefig(f"/Users/tatsumiryou/Spin_picture/Duffing/duffing_phase_f={self.Amp}.pdf")
        plt.show()

        plt.plot(self.t_eval,ans[0])
        ax1.spines["right"].set_color("none")  # グラフ右側の軸(枠)を消す
        ax1.spines["top"].set_color("none")  # グラフ上側の軸(枠)を消す
        ax1.spines['left'].set_position('zero')  # グラフ左側の軸(枠)が原点を通る
        ax1.spines['bottom'].set_position('zero')  # グラフ下側の軸(枠)が原点を通る

        plt.yticks([1.00, 0.50, 0, -0.50, -1])
        plt.grid(axis="y")
        plt.ylim(-1.0,1.0)
        plt.xlabel("t")
        plt.ylabel("x")
        plt.xlim(100,150)
        plt.savefig(f"/Users/tatsumiryou/Spin_picture/Duffing/duffing_tx_f={self.Amp}.pdf")
        plt.show()

        N = len(ans[0][300000:])  # サンプル数
        f_s = 400  # サンプリングレート f_s[Hz] (任意)
        dt = (self.t[1] - self.t[0]) / len(self.t_eval)  # サンプリング周期 dt[s]

        y_fft = np.fft.fft(ans[0][300000:])  # 離散フーリエ変換
        freq = np.fft.fftfreq(N, d=dt)  # 周波数を割り当てる（※後述）
        Amp = abs(y_fft / (N / 2))  # 音の大きさ（振幅の大きさ）
        plt.plot(freq[1:int(N / 2)], Amp[1:int(N / 2)])  # A-f グラフのプロット
        # plt.xscale("log")  # 横軸を対数軸にセット
        max_y = max(Amp[1:int(N / 2)])
        plt.xlim(0,5)
        plt.ylim(0,max_y + 0.01)
        plt.xlabel("GHz")
        #plt.savefig(f"/Users/tatsumiryou/Spin_picture/Duffing/Fourier/duffing_xFourier_f={self.Amp}.pdf")
        plt.show()

        # ani.save("reverse_spin.gif",writer='imagemagick')

    def Lyapunov(self,pertu,step):
        dX0 = np.array(self.X0) + np.array(pertu)
        Sol0 = sc.integrate.solve_ivp(self.func, self.t, self.X0, t_eval=self.t_eval, atol=1e-12, rtol=1e-12)
        Solp = sc.integrate.solve_ivp(self.func, self.t, dX0, t_eval=self.t_eval, atol=1e-12, rtol=1e-12)
        ans0 = Sol0.y.T
        ansp = Solp.y.T

        print(ans0,ansp)
        dist0 = np.linalg.norm(ans0[0] - ansp[0])

        Lya_dt = int(len(self.t_eval)/step)
        Lya = 0

        for i in range(step):
            distance = np.linalg.norm(ans0[i * Lya_dt]- ansp[i* Lya_dt])
            Lya += np.log(distance)

        Lya_expo = (Lya - np.log(dist0))/step
        print(Lya_expo)

    def poincore(self,start,stop):
        his = self.history()
        x = self.Sol.y[0]
        t = self.t_eval
        dt = (self.t[1] - self.t[0]) / len(self.t_eval)  # サンプリング周期 dt[s]
        T = 2 * np.pi / self.omega
        storobo = int(T / dt)

        print(storobo)
        x_storobo = x[start:stop:storobo]
        print(x_storobo)

        return x_storobo


if __name__ == '__main__':
    t = [0,500]
    t_eval = np.linspace(*t, 6000000)
    duff = Duffing(1,32,176,4.8,3.5,t,t_eval,[0.42,0.0001,3.5])
    duff.doit()
    duff.Lyapunov([0,0,0.01],1000)
    #duff.poincore(300000,600000)
