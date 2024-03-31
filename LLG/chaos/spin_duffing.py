import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from New_Spin_Lyapunov import Lyapunov
from one_spin.FMR import FMR_spin
from one_spin.FMR_gif import FMR_gif

class Duffing():
    def __init__(self, gamma,Bx,Ky, B0,omega, t,  t_eval, X0):
        self.Bx = Bx
        self.Ky = Ky
        self.gamma = gamma
        self.B0 = B0
        self.omega = omega
        self.t = t
        self.X0 = X0
        self.t_eval = t_eval

    def func(self,t,X):
        sin = np.sin(X[0])
        cos = np.cos(X[0])
        dtdx = X[1]
        dtdy = - self.gamma * np.tan(X[1]) * X[1] * X[1] - self.gamma ** 2 *(self.Bx * self.Bx * sin * cos - self.Bx * self.Ky * sin * (cos - sin *sin) - self.Ky * sin**3 * cos + self.B0 * self.omega * np.sin(X[2]))
        dtdz = self.omega
        dtdfunc = [dtdx,dtdy,dtdz]
        return dtdfunc


    def doit(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        self.Sol = sc.integrate.solve_ivp(self.func,self.t, self.X0, t_eval=self.t_eval,atol=1e-12,rtol=1e-12)
        ans = self.Sol.y

        print(ans)
        plt.plot(ans[0][150000:],ans[1][150000:])
        plt.xlabel("φ")
        plt.ylabel("dφ/dt")
        #plt.savefig(f"/Users/tatsumiryou/Spin_picture/Duffing/Spin_xy_duffing_phase_f={self.B0}.pdf")
        plt.show()
        #dt = np.diff(ans[0],1)
        #plt.plot(ans[0][30000:], dt[29999:])
        #plt.show()
        plt.plot(self.t_eval,ans[0])
        ax1.spines["right"].set_color("none")  # グラフ右側の軸(枠)を消す
        ax1.spines["top"].set_color("none")  # グラフ上側の軸(枠)を消す
        ax1.spines['left'].set_position('zero')  # グラフ左側の軸(枠)が原点を通る
        ax1.spines['bottom'].set_position('zero')  # グラフ下側の軸(枠)が原点を通る

        plt.yticks([1.00, 0.50, 0, -0.50, -1])
        plt.grid(axis="y")
        plt.ylim(-0.7, 0.7)
        plt.xlabel("t")
        plt.ylabel("φ")
        #plt.savefig(f"/Users/tatsumiryou/Spin_picture/Duffing/Spin_xy_duffing_tphi_f={self.B0}.pdf")
        plt.show()
        # ani.save("reverse_spin.gif",writer='imagemagick')

if __name__ == '__main__':
    t = [0,200]
    t_eval = np.linspace(*t, 200000)
    duff = Duffing(0.17,160,200,100,5,t,t_eval,[0.44,0.001,5])
    duff.doit()
