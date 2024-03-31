import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class Tool():
    def __init__(self,t,t_eval):
        self.t = t
        self.t_eval = t_eval

    def get_spin_vec(self,t):
        Ox,Oy,Oz = 0,0,0
        x = self.S[0][t]
        y = self.S[1][t]
        z = self.S[2][t]
        return Ox, Oy, Oz, x, y, z


    def update(self,t):
        self.quiveraa.remove()
        self.quiveraa = self.ax.quiver(*self.get_spin_vec(t))

    def doit(self):
        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection="3d"))

        self.Sol = sc.integrate.solve_ivp(self.func_S,self.t, self.S0, t_eval=self.t_eval,atol=1e-12,rtol=1e-12)
        self.S = self.Sol.y


        self.quiveraa = self.ax.quiver(*self.get_spin_vec(0))

        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(-2, 2)

        ani = FuncAnimation(self.fig, self.update, frames=len(self.Sol.t), interval=100)
        # ani.save("reverse_spin.gif",writer='imagemagick')
        plt.show()

        return self.S

    def history(self):
        self.Sol = sc.integrate.solve_ivp(self.func_S, self.t, self.S0, t_eval=self.t_eval, atol=1e-12, rtol=1e-12)
        self.S = self.Sol.y
        return self.S


    def tSi_graph(self,start_time,end_time,i_list,font_size=20):
        t = self.Sol.t

        # plt.savefig("reverse_x.png")
        for i in i_list:
            ax_name =''
            if i == 0:
                ax_name = "x"
            elif i == 1:
                ax_name = "y"
            elif i == 2:
                ax_name = "z"
            Si = self.S[i]  # i=[0]ならSx,i=[1]ならSy,i=[0,2]ならSxとSz
            plt.plot(t,Si,label = "S" + ax_name)
        plt.xlim(start_time, end_time)
        plt.legend()
        plt.rcParams["font.size"] = font_size
        plt.show()


    def Si_fourier(self,start_step,end_step,ax,show_omega_range,font_size=20): #show_omega_rangeは二つ要素を持つ配列で見たい周波数幅を入れる
        t = self.Sol.t

        Si = self.S[ax]  # i=[0]ならSx,i=[1]ならSy
        N = len(Si[start_step:end_step])  # サンプル数
        f_s = 400  # サンプリングレート f_s[Hz] (任意)
        dt = (self.t[1] - self.t[0]) / len(self.t_eval)  # サンプリング周期 dt[s]

        y_fft = np.fft.fft(Si[start_step:end_step])  # 離散フーリエ変換
        freq = np.fft.fftfreq(N, d=dt)  # 周波数を割り当てる（※後述）
        Amp = abs(y_fft / (N / 2))  # 音の大きさ（振幅の大きさ）
        plt.plot(freq[1:int(N / 2)], Amp[1:int(N / 2)])  # A-f グラフのプロット
        # plt.xscale("log")  # 横軸を対数軸にセット
        plt.xlim(show_omega_range[0], show_omega_range[1])
        plt.xlabel("GHz")

        plt.show()

    def poincore(self,ax, start, stop):
        Si = self.Sol.y[ax]
        t = self.t_eval
        dt = (self.t[1] - self.t[0]) / len(self.t_eval)  # サンプリング周期 dt[s]
        T = 2 * np.pi / self.omega[ax]
        storobo = int(T / dt)

        print(storobo)
        Si_storobo = Si[start:stop:storobo]
        print(Si_storobo)

        return Si_storobo

    def get_spin_vec(self,t):
        Ox, Oy, Oz = 0, 0, 0
        x = self.S[0][t]
        y = self.S[1][t]
        z = self.S[2][t]
        self.ax.plot(x, y, z, marker='o', markersize=2.5, color='b')
        return Ox, Oy, Oz, x, y, z

    def make_gif(self):
        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection="3d"))

        self.quiveraa = self.ax.quiver(*self.get_spin_vec(0))

        r  = np.linalg.norm(self.S0) # 半径を指定
        theta_1_0 = np.linspace(0, np.pi * 2, 100)  # θ_1は&#91;0,π/2]の値をとる
        theta_2_0 = np.linspace(0, np.pi, 100)  # θ_2は&#91;0,π/2]の値をとる
        theta_1, theta_2 = np.meshgrid(theta_1_0, theta_2_0)  # ２次元配列に変換
        x = np.cos(theta_2) * np.sin(theta_1) * r  # xの極座標表示
        y = np.sin(theta_2) * np.sin(theta_1) * r  # yの極座標表示
        z = np.cos(theta_1) * r  # zの極座標表示

        self.ax.plot_surface(x, y, z, alpha=0.2)  # 球を３次元空間に表示

        """
        self.ax.quiver(self.plotB[0][0], self.plotB[0][1], self.plotB[0][2], self.plotB[1][0], self.plotB[1][1], self.plotB[1][2], color='k', arrow_length_ratio=0.18, linewidth=3, label='B')
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.text(0.12, 0.15, -0.3, "B",size=20, )
        """

        self.ax.view_init(elev=20, azim=20)
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_zlim(-1.2, 1.2)
        self.ax.set_aspect('equal')
        #self.ax.view_init(azim=1,elev=88)


        ani = FuncAnimation(self.fig, self.update, frames=len(self.Sol.t), interval=1)
        #ani.save("no_dunp_spin_10fps.gif", fps = 10)
        #ani.save('pillow_imagedraw.gif', duration=40, loop=0)
        plt.show()

    def make_trajectory(self,start_step,end_step):
        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection="3d"))

        #self.quiveraa = self.ax.quiver(*self.get_spin_vec(0))

        r = np.linalg.norm(self.S0)  # 半径を指定
        theta_1_0 = np.linspace(0, np.pi * 2, 100)  # θ_1は&#91;0,π/2]の値をとる
        theta_2_0 = np.linspace(0, np.pi, 100)  # θ_2は&#91;0,π/2]の値をとる
        theta_1, theta_2 = np.meshgrid(theta_1_0, theta_2_0)  # ２次元配列に変換
        x = np.cos(theta_2) * np.sin(theta_1) * r  # xの極座標表示
        y = np.sin(theta_2) * np.sin(theta_1) * r  # yの極座標表示
        z = np.cos(theta_1) * r  # zの極座標表示

        self.ax.plot_surface(x, y, z, alpha=0.2)  # 球を３次元空間に表示

        S = self.S
        self.ax.scatter(S[0][start_step:end_step], S[1][start_step:end_step], S[2][start_step:end_step], s=1, c="blue")

        # self.ax.quiver(self.plotB[0][0], self.plotB[0][1], self.plotB[0][2], self.plotB[1][0], self.plotB[1][1], self.plotB[1][2], color='k', arrow_length_ratio=0.18, linewidth=3, label='B')
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.text(0.12, 0.15, -0.3, "B", size=20, )
        #print(1)

        self.ax.view_init(elev=20, azim=20)
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_zlim(-1.2, 1.2)
        self.ax.set_aspect('equal')
        # self.ax.view_init(azim=1,elev=88)

        plt.show()


