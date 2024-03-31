import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from chaos.FMR_duffing import Duffing
from matplotlib.animation import FuncAnimation

class PloAni(Duffing):
    def __init__(self,alpha, gamma,Bx,Ky, B0,omega, t,  t_eval, S0, STO):
        super().__init__(alpha, gamma,Bx,Ky, B0,omega, t,  t_eval, S0, STO)
        self.his = self.history()

    def make_2D(self):
        t = self.his[4]
        x = np.sin(self.his[0]) * np.cos(self.his[2])
        y = np.sin(self.his[0]) * np.sin(self.his[2])
        z = np.cos(self.his[0])
        plt.plot(t, x, label="Sx")
        plt.ylabel("Sx")
        plt.xlabel("t(ns)")
        plt.ylim(-1.0, 1.0)
        # plt.savefig("reverse_x.png")
        plt.show()

        N = len(x)  # サンプル数
        f_s = 400  # サンプリングレート f_s[Hz] (任意)
        dt = (self.t[1] - self.t[0]) / len(self.t_eval)  # サンプリング周期 dt[s]

        y_fft = np.fft.fft(x)  # 離散フーリエ変換
        freq = np.fft.fftfreq(N, d=dt)  # 周波数を割り当てる（※後述）
        Amp = abs(y_fft / (N / 2))  # 音の大きさ（振幅の大きさ）
        plt.plot(freq[1:int(N / 2)], Amp[1:int(N / 2)])  # A-f グラフのプロット
        # plt.xscale("log")  # 横軸を対数軸にセット
        plt.xlim(0, 20)
        plt.xlabel("GHz")
        plt.show()



        plt.plot(t, y, label="Sy")
        plt.ylabel("Sy")
        plt.xlabel("t(ns)")
        plt.ylim(-1.0, 1.0)
        # plt.savefig("reverse_x.png")
        plt.show()



        plt.plot(t, z, label="Sz")
        # plt.savefig("reverse_x.png")
        plt.ylabel("Sz")
        plt.xlabel("t(ns)")
        plt.ylim(-1.0, 1.0)
        plt.show()

        N = len(z)  # サンプル数
        f_s = 400  # サンプリングレート f_s[Hz] (任意)
        dt = (self.t[1] - self.t[0]) / len(self.t_eval)  # サンプリング周期 dt[s]

        y_fft = np.fft.fft(z)  # 離散フーリエ変換
        freq = np.fft.fftfreq(N, d=dt)  # 周波数を割り当てる（※後述）
        Amp = abs(y_fft / (N / 2))  # 音の大きさ（振幅の大きさ）
        plt.plot(freq[1:int(N / 2)], Amp[1:int(N / 2)])  # A-f グラフのプロット
        # plt.xscale("log")  # 横軸を対数軸にセット
        # plt.xlim(0, 20)
        plt.xlabel("GHz")
        plt.xlim(0, 20)
        plt.show()

        N = len(z[100000:])  # サンプル数
        f_s = 400  # サンプリングレート f_s[Hz] (任意)
        dt = (self.t[1] - self.t[0]) / len(self.t_eval)  # サンプリング周期 dt[s]

        y_fft = np.fft.fft(z[100000:])  # 離散フーリエ変換
        freq = np.fft.fftfreq(N, d=dt)  # 周波数を割り当てる（※後述）
        Amp = abs(y_fft / (N / 2))  # 音の大きさ（振幅の大きさ）
        plt.plot(freq[1:int(N / 2)], Amp[1:int(N / 2)])  # A-f グラフのプロット
        # plt.xscale("log")  # 横軸を対数軸にセット
        # plt.xlim(0, 20)
        plt.xlabel("GHz")
        plt.xlim(0, 20)
        plt.show()


    def get_spin_vec(self,t):
        Ox, Oy, Oz = 0, 0, 0
        x = np.sin(self.his[0][t]) * np.cos(self.his[2][t])
        y = np.sin(self.his[0][t]) * np.sin(self.his[2][t])
        z = np.cos(self.his[0][t])
        self.ax.plot(x, y, z, marker='o', markersize=2.5, color='b')
        return Ox, Oy, Oz, x, y, z

    def update(self,t):
        self.quiveraa.remove()
        self.quiveraa = self.ax.quiver(*self.get_spin_vec(t))

    def make_Ani(self):
        his = self.history()
        t = his[4]
        x = np.sin(his[0]) * np.cos(his[2])
        y = np.sin(his[0]) * np.sin(his[2])
        z = np.cos(his[0])

        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection="3d"))


        self.quiveraa = self.ax.quiver(*self.get_spin_vec(0))

        r = 1  # 半径を指定
        theta_1_0 = np.linspace(0, np.pi * 2, 100)  # θ_1は&#91;0,π/2]の値をとる
        theta_2_0 = np.linspace(0, np.pi, 100)  # θ_2は&#91;0,π/2]の値をとる
        theta_1, theta_2 = np.meshgrid(theta_1_0, theta_2_0)  # ２次元配列に変換
        x = np.cos(theta_2) * np.sin(theta_1) * r  # xの極座標表示
        y = np.sin(theta_2) * np.sin(theta_1) * r  # yの極座標表示
        z = np.cos(theta_1) * r  # zの極座標表示

        self.ax.plot_surface(x, y, z, alpha=0.2)  # 球を３次元空間に表示

        # self.ax.quiver(self.plotB[0][0], self.plotB[0][1], self.plotB[0][2], self.plotB[1][0], self.plotB[1][1], self.plotB[1][2], color='k', arrow_length_ratio=0.18, linewidth=3, label='B')
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.text(0.12, 0.15, -0.3, "B", size=20, )

        self.ax.view_init(elev=20, azim=20)
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_zlim(-1.2, 1.2)
        self.ax.set_aspect('equal')
        # self.ax.view_init(azim=1,elev=88)

        ani = FuncAnimation(self.fig, self.update, frames=len(t), interval=1)
        # ani.save(f"STO_{self.omega[1]}GHz_{self.STO_ac_effH[1]}Adiv^2.gif", fps = 40)
        # ani.save('pillow_imagedraw.gif', duration=40, loop=0)
        plt.show()

    def poincare(self):
        t = self.his[4]
        dt = (self.t[1] - self.t[0]) / len(self.t_eval)  # サンプリング周期 dt[s]
        T = 2 * np.pi / self.omega
        storobo = int(T / dt)

        print(storobo)
        theta = self.his[0][::storobo]
        dtheta = self.his[1][::storobo]
        phi = self.his[2][::storobo]
        dphi = self.his[3][::storobo]
        plt.plot(theta, dtheta, label="theta_poin")
        # plt.savefig("reverse_x.png")
        plt.show()

        plt.plot(phi, dphi, label="phi_poin")
        # plt.savefig("reverse_x.png")
        plt.show()

    def Fourier(self):
        plt.clf()
        ans = self.history()
        theta = ans[0]
        phi = ans[2]
        N = len(phi)  # サンプル数
        f_s = 400  # サンプリングレート f_s[Hz] (任意)
        dt = (self.t[1] - self.t[0]) / len(self.t_eval)  # サンプリング周期 dt[s]

        y_fft = np.fft.fft(phi)  # 離散フーリエ変換
        freq = np.fft.fftfreq(N, d=dt)  # 周波数を割り当てる（※後述）
        Amp = abs(y_fft / (N / 2))  # 音の大きさ（振幅の大きさ）
        plt.plot(freq[1:int(N / 2)], Amp[1:int(N / 2)])  # A-f グラフのプロット
        # plt.xscale("log")  # 横軸を対数軸にセット
        plt.xlim(0,20)
        plt.xlabel("GHz")
        plt.show()

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
        for i in x_storobo:
            print(i)



if __name__ == '__main__':
    t = [0,3000]
    t_eval = np.linspace(*t, 200000)
    duff = PloAni(0.05,0.17,165,200,10.5,20.2244,t,t_eval,[np.pi/2,0.6435,0],0.1)
    duff.doit()
    duff.poincore(100000,200000)
    #duff.make_2D()
    #duff.Fourier()
    #duff.make_Ani()
    #duff.poincare()
