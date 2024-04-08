import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class Tool():
    def __init__(self,t,t_eval):
        self.t = t
        self.t_eval = t_eval

    def history(self):
        self.Sol = sc.integrate.solve_ivp(self.func, self.t, self.X0, t_eval=self.t_eval, atol=1e-12, rtol=1e-12)
        return self.Sol

    def phase_graph(self,x,y,x_name,y_name,start_step,end_step): #x,yは相図の横軸および縦軸にする配列のIndex
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ans = self.Sol.y
        plt.plot(ans[x][start_step:end_step], ans[y][start_step:end_step])
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        # plt.savefig(f"/Users/tatsumiryou/Spin_picture/Duffing/duffing_phase_f={self.Amp}.pdf")
        plt.show()

    def diff_phase_graph(self,x,x_name,dxdt_name,start_step,end_step):
        ans = self.Sol.y
        dxdt = np.diff(ans[x], 1)
        plt.plot(ans[x][start_step:end_step], dxdt[start_step-1 : end_step-1])
        plt.xlabel(x_name)
        plt.ylabel(dxdt_name)
        # plt.savefig(f"/Users/tatsumiryou/Spin_picture/Duffing/duffing_phase_f={self.Amp}.pdf")
        plt.show()


    def tft_graph(self,y,y_name,start_time,end_time):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ans = self.Sol.y
        plt.plot(self.t_eval, ans[y])
        ax1.spines["right"].set_color("none")  # グラフ右側の軸(枠)を消す
        ax1.spines["top"].set_color("none")  # グラフ上側の軸(枠)を消す
        ax1.spines['left'].set_position('zero')  # グラフ左側の軸(枠)が原点を通る
        ax1.spines['bottom'].set_position('zero')  # グラフ下側の軸(枠)が原点を通る

        # plt.yticks([1.00, 0.50, 0, -0.50, -1])
        plt.xlim(start_time,end_time)
        plt.xlabel("t")
        plt.ylabel(y_name)
        # plt.xlim(100,150)
        # plt.savefig(f"/Users/tatsumiryou/Spin_picture/Duffing/duffing_tx_f={self.Amp}.pdf")
        plt.show()

    def fourier(self,ax,ax_name,start_step,end_step,show_omega_range):
        ans = self.Sol.y[ax] # i=[0]ならSx,i=[1]ならSy
        N = len(ans[start_step:end_step])  # サンプル数
        f_s = 400  # サンプリングレート f_s[Hz] (任意)
        dt = (self.t[1] - self.t[0]) / len(self.t_eval)  # サンプリング周期 dt[s]

        y_fft = np.fft.fft(ans[start_step:end_step])  # 離散フーリエ変換
        freq = np.fft.fftfreq(N, d=dt)  # 周波数を割り当てる（※後述）
        Amp = abs(y_fft / (N / 2))  # 音の大きさ（振幅の大きさ）
        plt.plot(freq[1:int(N / 2)], Amp[1:int(N / 2)])  # A-f グラフのプロット
        # plt.xscale("log")  # 横軸を対数軸にセット
        plt.xlim(show_omega_range[0], show_omega_range[1])
        plt.xlabel("GHz")

        plt.show()

    def ani_history(self):
        self.Sol = sc.integrate.solve_ivp(self.func, self.t, self.S0, t_eval=self.t_eval,atol=1e-12,rtol=1e-12)
        ans = self.Sol.y
        t = self.Sol.t
        dtdth = np.diff(ans[0], 1)
        dtdph = np.diff(ans[1], 1)
        return [ans[0], dtdth, ans[1], dtdph, t]

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
        self.his = self.ani_history()
        t = self.his[4]
        x = np.sin(self.his[0]) * np.cos(self.his[2])
        y = np.sin(self.his[0]) * np.sin(self.his[2])
        z = np.cos(self.his[0])

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

    def Lyapunov(self,pertu,step):
        dX0 = np.array(self.X0) + np.array(pertu)
        Sol0 = sc.integrate.solve_ivp(self.func, self.t, self.X0, t_eval=self.t_eval, atol=1e-12, rtol=1e-12)
        Solp = sc.integrate.solve_ivp(self.func, self.t, dX0, t_eval=self.t_eval, atol=1e-12, rtol=1e-12)
        ans0 = Sol0.y.T
        ansp = Solp.y.T

        dist0 = np.linalg.norm(ans0[0] - ansp[0])

        Lya_dt = int(len(self.t_eval)/step)
        Lya = 0

        for i in range(step):
            distance = np.linalg.norm(ans0[i * Lya_dt]- ansp[i* Lya_dt])
            Lya += np.log(distance)
            #print(Lya)

        Lya_expo = (Lya)/step - np.log(dist0)
        print("here",Lya_expo)

    def matsunaga_Lyapunov(self, pertu, step, cal_num, start_step):
        dt = (self.t[1] - self.t[0]) / len(self.t_eval)
        Lya_dt = int((len(self.t_eval) - start_step) / step)
        Sol0 = sc.integrate.solve_ivp(self.func, self.t, self.X0, t_eval=self.t_eval, atol=1e-12, rtol=1e-12)
        ans0 = Sol0.y.T
        # print(ans0)
        # print(np.array(ans0[start_step]) + np.array(pertu))

        dX0 = np.array(ans0[start_step]) + np.array(pertu)
        t_evalp = np.linspace(*[0, cal_num * dt * Lya_dt], Lya_dt * cal_num)
        Solp = sc.integrate.solve_ivp(self.func, [0, cal_num * dt * Lya_dt], dX0, t_eval=t_evalp, atol=1e-12,
                                      rtol=1e-12)
        ansp = Solp.y.T

        # print(ans0, ansp)
        dist0 = np.linalg.norm(ans0[start_step][:-1] - ansp[0][:-1])
        print(dist0)

        Lya = 0

        for i in range(step):
            # print(len(ansp), Lya_dt,(i + 1) * Lya_dt + start_step,len(ans0))
            # print(ansp[Lya_dt] ,ans0[(i + 1) * Lya_dt + start_step])
            ansp[Lya_dt][-1] = ans0[(i + 1) * Lya_dt + start_step][-1]
            pi = np.linalg.norm(ans0[(i + 1) * Lya_dt + start_step] - ansp[Lya_dt]) / dist0
            # print(i,pi)
            # print(ans0[(i + 1) * Lya_dt], ansp[Lya_dt])
            per_X0i = ans0[(i + 1) * Lya_dt + start_step] + (ansp[Lya_dt] - ans0[(i + 1) * Lya_dt + start_step]) / pi
            # per_X0i[-1] = (i + 1) * dt * Lya_dt * self.omega
            # print(t_evalp)

            tp = [0, cal_num * dt * Lya_dt]

            t_evalp = np.linspace(*[0, cal_num * dt * Lya_dt], Lya_dt * cal_num)
            # print(tp,t_evalp)
            Solp = sc.integrate.solve_ivp(self.func, tp, per_X0i, t_eval=t_evalp, atol=1e-12, rtol=1e-12)
            ansp = Solp.y.T
            Lya += np.log(pi)
            print(Lya)

        cal_time = self.t[1] * (Lya_dt * step / len(self.t_eval))
        # print(cal_time)
        Lyap_expo = Lya / cal_time
        print(Lyap_expo)
        return Lyap_expo


    def poincore(self,start,stop):
        his = self.history()
        x = self.Sol.y[0]
        t = self.t_eval
        dt = (self.t[1] - self.t[0]) / len(self.t_eval)  # サンプリング周期 dt[s]
        T = 2 * np.pi / self.omega
        storobo = int(T / dt)

        #print(storobo)
        x_storobo = x[start:stop:storobo]
        #print(x_storobo)

        return x_storobo

    def FMR_matsunaga_Lyapunov(self,pertu,step,cal_rate):
        dt = (self.t[1] - self.t[0]) / len(self.t_eval)
        Lya_dt = int(len(self.t_eval) / step)
        dX0 = np.array(self.S0) + np.array(pertu)
        Sol0 = sc.integrate.solve_ivp(self.func, self.t, self.X0, t_eval=self.t_eval, atol=1e-12, rtol=1e-12)
        t_evalp = np.linspace(*[0, 10 * dt * Lya_dt], Lya_dt * 10)
        #Solp = sc.integrate.solve_ivp(self.func, [0, 10 * dt * Lya_dt], dX0, t_eval=t_evalp, atol=1e-12, rtol=1e-12)
        Solp = sc.integrate.solve_ivp(self.func, self.t, dX0, t_eval=self.t_eval, atol=1e-12, rtol=1e-12)
        ans0 = Sol0.y.T
        ansp = Solp.y.T

        #print(ans0, ansp)
        dist0 = np.linalg.norm(ans0[0] - ansp[0])

        Lya = 0

        for i in range(step - 1):
            ansp[Lya_dt][-1] = ans0[(i + 1) * Lya_dt][-1]
            pi = np.linalg.norm(ans0[(i + 1) * Lya_dt] - ansp[Lya_dt]) / dist0
            #print(i,pi)
            #print(ans0[(i + 1) * Lya_dt],ansp[Lya_dt])
            per_X0i = ans0[(i + 1) * Lya_dt] + (ansp[Lya_dt] - ans0[(i + 1) * Lya_dt])/pi
            #per_X0i[-1] = (i + 1) * dt * Lya_dt * self.omega
            tp = [self.t[0],self.t[1] * cal_rate]
            #print(tp)
            eval_len = int((len(self.t_eval) - 1) * cal_rate + 1)
            #print(eval_len)
            t_evalp = np.linspace(*tp,eval_len)
            #print(tp,t_evalp)
            Solp = sc.integrate.solve_ivp(self.func, tp, per_X0i, t_eval = t_evalp, atol=1e-12, rtol=1e-12)
            ansp = Solp.y.T
            Lya += np.log(pi)
            #print(Lya)

        Lyap_expo = Lya/self.t[1]
        print(Lyap_expo)
        return Lyap_expo

