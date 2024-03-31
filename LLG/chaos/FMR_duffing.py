import scipy as sc
import numpy as np
import matplotlib.pyplot as plt


class Duffing():
    def __init__(self, alpha, gamma,Bx,Ky, B0,omega, t,  t_eval, S0, STO):
        self.alpha = alpha
        self.Bx = Bx
        self.Ky = Ky
        self.gamma = gamma
        self.B0 = B0
        self.omega = omega
        self.t = t
        self.S0 = S0
        self.t_eval = t_eval
        self.STO = STO
        self.dt = []

    def func(self,t,S):
        #print(S)
        sinth = np.sin(S[0])
        costh = np.cos(S[0])
        sinph = np.sin(S[1])
        cosph = np.cos(S[1])
        dtdth = self.gamma * (-self.Bx * sinph + self.Ky  * sinth * sinph * cosph  + self.B0  * cosph * np.sin(S[2])) + self.alpha * self.gamma * (self.Bx * costh * cosph  + self.Ky * sinth * costh * sinph * sinph + self.B0  * sinph * np.sin(S[2]) * costh)
        dtdph = self.gamma * (-self.Bx * costh * cosph / (sinth) - self.Ky * costh * sinph * sinph - self.B0  * sinph * np.sin(S[2]) * costh/ sinth) + self.alpha * self.gamma * (-self.Bx * sinph / sinth + self.Ky  * sinph * cosph  + self.B0  * cosph * np.sin(S[2])/sinth)
        dtdo = self.omega
        dtdfunc = [dtdth,dtdph, dtdo]
        #print(dtdfunc)
        #print(t)
        return dtdfunc

    def doit(self):
        self.Sol = sc.integrate.solve_ivp(self.func, self.t, self.S0, t_eval= self.t_eval,atol=1e-12,rtol=1e-12)
        print(self.t,self.t_eval)
        ans = self.Sol.y
        print(ans)
        dtdth = np.diff(ans[0], 1)
        dtdph = np.diff(ans[1], 1)

        plt.plot(ans[0][190000:], dtdth[189999:])
        plt.xlabel("θ(rad)")
        plt.ylabel("dθ/dt")
        #plt.savefig(f"/Users/tatsumiryou/Spin_picture/Duffing/Spin_duffing_theta_phase_f={self.B0}_alpha={self.alpha}.pdf")
        plt.show()
        plt.plot(self.t_eval, ans[0])
        plt.ylabel("θ(rad)")
        plt.xlabel("t(ns)")
        #plt.savefig(f"/Users/tatsumiryou/Spin_picture/Duffing/Spin_duffing_theta_ttheta_f={self.B0}_alpha={self.alpha}.pdf")
        plt.show()

        plt.plot(ans[1][190000:], dtdph[189999:])
        plt.xlabel("φ(rad)")
        plt.ylabel("dφ/dt")
        plt.savefig(f"/Users/tatsumiryou/Spin_picture/Duffing/Spin_duffing_phi_phase_f={self.B0}_alpha={self.alpha}.pdf")
        plt.show()
        plt.plot(self.t_eval,ans[1])
        plt.ylabel("φ(rad)")
        plt.xlabel("t(ns)")
        plt.ylim(-1.0, 1.0)
        plt.xlim(200, 250)
        plt.savefig(f"/Users/tatsumiryou/Spin_picture/Duffing/Spin_duffing_phi_tphi_f={self.B0}_alpha={self.alpha}.pdf")
        plt.show()
        # ani.save("reverse_spin.gif",writer='imagemagick')

    def history(self):
        self.Sol = sc.integrate.solve_ivp(self.func, self.t, self.S0, t_eval=self.t_eval,atol=1e-12,rtol=1e-12)
        ans = self.Sol.y
        t = self.Sol.t
        dtdth = np.diff(ans[0], 1)
        dtdph = np.diff(ans[1], 1)
        return [ans[0], dtdth, ans[1], dtdph, t]

    def Lyapunov(self,pertu,step):
        dX0 = np.array(self.S0) + np.array(pertu)
        Sol0 = sc.integrate.solve_ivp(self.func, self.t, self.S0, t_eval=self.t_eval, atol=1e-12, rtol=1e-12)
        ans0 = Sol0.y
        dtdth0 = np.diff(ans0[0], 1)
        dtdph0 = np.diff(ans0[1], 1)
        ans0a = np.array([ans0[0][:-1], dtdth0, ans0[1][:-1], dtdph0]).T
        print(ans0a)

        Solp = sc.integrate.solve_ivp(self.func, self.t, dX0, t_eval=self.t_eval, atol=1e-12, rtol=1e-12)

        ansp = Solp.y
        dtdthp = np.diff(ansp[0], 1)
        dtdphp = np.diff(ansp[1], 1)
        anspa = np.array([ansp[0][:-1], dtdthp, ansp[1][:-1], dtdphp]).T


        print(ans0a,anspa)
        dist0 = np.linalg.norm(ans0a[0] - anspa[0])
        print(dist0)

        Lya_dt = int(len(self.t_eval)/step)
        Lya = 0

        for i in range(step):
            print(i)
            distance = np.linalg.norm(ans0a[i * Lya_dt]- anspa[i* Lya_dt])
            print(distance)
            Lya += np.log(distance)

        Lya_expo = Lya/step - np.log(dist0)
        print(Lya_expo)

if __name__ == '__main__':
    t = [0,300]
    t_eval = np.linspace(*t, 200000)
    duff = Duffing(0.05,0.17,165,200,10.7,20.2244,t,t_eval,[np.pi/2,0.6435,0],0.1)
    duff.doit()
    #duff.Lyapunov([0.001,0.001,0],1000)