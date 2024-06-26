import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation
from FMR import FMR

class FMR_Phase_space(FMR):
    def __init__(self,alpha,edge_alpha,gamma,B,S0,t,t_eval,Kx,Ky,Kz,beta,Amp,omega,N,J,start,stop):
        super().__init__(alpha,edge_alpha,gamma,B,S0,t,t_eval,Kx,Ky,Kz,beta,Amp,omega,N,J,start,stop)

    def solving(self):
        self.S0 = np.reshape(self.S0, (1, -1))
        self.S0 = list(self.S0[0])
        # self.fig, self.ax = plt.subplots(subplot_kw=dict(projection="3d"))
        print(self.S0)

        self.Sol = sc.integrate.solve_ivp(self.func_S, self.t, self.S0, t_eval=self.t_eval,rtol=1e-6, atol=1e-9)
        self.S = self.Sol.y
        self.spin_log = np.reshape(self.S, (self.N, 3, -1))


    def make_phase_graph(self,i,ax1,ax2):      #二次元のぐらふを作る。iはi番目のスピン、axはどの軸をとるか
        t = self.t_eval
        x = self.spin_log[i][0]
        y = self.spin_log[i][1]
        z = self.spin_log[i][2]

        plt.plot(x,y)
        plt.xlabel("Sx")
        plt.ylabel("Sy")
        plt.savefig(f"FMR_84mT_SxSy_{i+1}.png")
        plt.show()

        """

        plt.plot(x, z)
        plt.xlabel("Sx")
        plt.ylabel("Sz")
        plt.show()

        plt.plot(z, y)
        plt.xlabel("Sz")
        plt.ylabel("Sy")
        plt.show()
        """

        plt.plot(t, x)
        plt.xlabel("t")
        plt.ylabel("Sx")
        plt.savefig(f"FMR_84mT_tSx_{i + 1}.png")
        plt.show()

        N = len(x)  # サンプル数
        f_s = 400  # サンプリングレート f_s[Hz] (任意)
        dt = (self.t[1] - self.t[0]) / len(self.t_eval)  # サンプリング周期 dt[s]

        y_fft = np.fft.fft(x)  # 離散フーリエ変換
        freq = np.fft.fftfreq(N, d=dt)  # 周波数を割り当てる（※後述）
        Amp = abs(y_fft / (N / 2))  # 音の大きさ（振幅の大きさ）
        plt.plot(freq[1:int(N / 2)], Amp[1:int(N / 2)])  # A-f グラフのプロット
        #plt.xscale("log")  # 横軸を対数軸にセット
        plt.xlabel("GHz")
        plt.savefig(f"FMR_84mT_Fourier_{i + 1}.png")

        plt.show()
        """

        plt.plot(t, z)
        plt.xlabel("t")
        plt.ylabel("Sz")
        plt.show()

        N = len(z)  # サンプル数
        f_s = 400  # サンプリングレート f_s[Hz] (任意)
        dt = (self.t[1] - self.t[0]) / len(self.t_eval)  # サンプリング周期 dt[s]

        y_fft = np.fft.fft(z)  # 離散フーリエ変換
        freq = np.fft.fftfreq(N, d=dt)  # 周波数を割り当てる（※後述）
        Amp = abs(y_fft / (N / 2))  # 音の大きさ（振幅の大きさ）
        plt.plot(freq[1:int(N / 2)], Amp[1:int(N / 2)])  # A-f グラフのプロット
        #plt.xscale("log")  # 横軸を対数軸にセット

        plt.show()
        #plt.show()
        """

    def make_3d(self,i):
        t = self.t_eval
        x = self.spin_log[i][0]
        y = self.spin_log[i][1]
        z = self.spin_log[i][2]
        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection="3d"))
        self.ax.plot(x, y, z)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        plt.show()

if __name__ == '__main__':
        n = 50
        S0 = np.zeros((n, 3))

        for i in range(n):
            S0[i][0] = 0.9571
            S0[i][2] = 0.259

        t = [0, 40]  # t(時間)が0〜100まで動き、その時のfを求める。
        t_eval = np.linspace(*t, 20000)

        mu_0 = 1.2
        gamma = 0.17
        h_div_2e = [0.329, -15]
        sta_M = [1.4, 0]  # 飽和磁化(T)で入れる
        theta = [-2.2, -1]
        j = [4.5, 9]
        d = [1.48, -9]
        Hsn = h_div_2e[0] * theta[0] * j[0] / (sta_M[0] * d[0])
        Hso = h_div_2e[1] + theta[1] + j[1] - (sta_M[1] + d[1])
        Hs = Hsn * (10 ** Hso) * 1000 * (mu_0 / 1200000) * mu_0  # 最後の1000はmTにするため

        """
        sinspin = Sin_Phase_space(0.0001, 0.1, gamma, [0, 0, mu_0 * 12], S0, t, t_eval, mu_0 * 4, 0, - mu_0 * 41.6, 0,
                             [0, Hs, 0], [0, 1, 0], n, 1, 0, 100)
        sinspin.solving()
        sinspin.make_phase_graph(24, 1, 2)
        sinspin.make_phase_graph(49, 1, 2)
        sinspin.make_phase_graph(99, 1, 2)
        sinspin.make_3d(24)
        sinspin.make_3d(49)
        sinspin.make_3d(99)
        """
        S0 = np.zeros((n, 3))

        for i in range(n):
            S0[i][2] = 1

        print(Hs)

        spin = FMR_Phase_space(0, 0, gamma, [0, 0, -mu_0 * 12], S0, t, t_eval, 0, 0, mu_0 * 220, 0,
                               [Hs,Hs,0],[47,47,0], n, 10, 0, 100)

        spin.solving()
        spin.make_phase_graph(0, 1, 2)
        spin.make_phase_graph(24, 1, 2)
        spin.make_phase_graph(49, 1, 2)
        #spin.make_phase_graph(99, 1, 2)
        spin.make_3d(0)
        spin.make_3d(24)
        spin.make_3d(49)
        #spin.make_3d(99)