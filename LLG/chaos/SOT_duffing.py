import scipy as sc
import numpy as np
import matplotlib.pyplot as plt


class SOT_Duffing():
    def __init__(self, alpha, gamma,Bx,Ky, SOT,omega, t,  t_eval, S0, STO):
        self.alpha = alpha
        self.Bx = Bx
        self.Ky = Ky
        self.gamma = gamma
        self.SOT = SOT #磁場換算した場合のSOTの振幅(gamma * SOTでトルクとなる)
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
        dtdth = self.gamma * (-self.Bx * sinph + self.Ky  * sinth * sinph * cosph  + self.SOT * np.sin(S[2]) * costh * sinth) + self.alpha * self.gamma * (self.Bx * costh * cosph  + self.Ky * sinth * costh * sinph * sinph)
        dtdph = self.gamma * (-self.Bx * costh * cosph / (sinth) - self.Ky * costh * sinph * sinph + self.SOT  *  np.sin(S[2]) * cosph/ sinth) + self.alpha * self.gamma * (-self.Bx * sinph / sinth + self.Ky * sinph * cosph)
        dtdo = self.omega
        dtdfunc = [dtdth,dtdph, dtdo]
        #print(dtdfunc)
        #print(t)
        return dtdfunc

    def doit(self):
        self.Sol = sc.integrate.solve_ivp(self.func, self.t, self.S0, t_eval= self.t_eval)
        print(self.t,self.t_eval)
        ans = self.Sol.y
        print(ans)
        dtdth = np.diff(ans[0], 1)
        dtdph = np.diff(ans[1], 1)

        plt.plot(ans[0][150000:], dtdth[149999:])
        plt.xlabel("theta")
        #plt.savefig(f"/Users/tatsumiryou/Spin_picture/Duffing/SOT_duffing_theta_phase_f={self.SOT}.pdf")
        plt.show()
        plt.plot(self.t_eval, ans[0])
        #plt.savefig(f"/Users/tatsumiryou/Spin_picture/Duffing/SOT_duffing_theta_ttheta_f={self.SOT}.pdf")
        plt.show()

        plt.plot(ans[1][150000:], dtdph[149999:])
        plt.xlabel("phi")
        #plt.savefig(f"/Users/tatsumiryou/Spin_picture/Duffing/SOT_duffing_phi_phase_f={self.SOT}.pdf")
        plt.show()
        plt.plot(self.t_eval,ans[1])
        #plt.savefig(f"/Users/tatsumiryou/Spin_picture/Duffing/SOT_duffing_phi_tphi_f={self.SOT}.pdf")
        plt.show()
        # ani.save("reverse_spin.gif",writer='imagemagick')

    def history(self):
        self.Sol = sc.integrate.solve_ivp(self.func, self.t, self.S0, t_eval=self.t_eval,atol=1e-12,rtol=1e-12)
        ans = self.Sol.y
        t = self.Sol.t
        dtdth = np.diff(ans[0], 1)
        dtdph = np.diff(ans[1], 1)
        return [ans[0], dtdth, ans[1], dtdph, t]

if __name__ == '__main__':
    t = [0,200]
    t_eval = np.linspace(*t, 200000)
    duff = SOT_Duffing(0.005,0.17,160,200,35,2,t,t_eval,[np.pi/2,0.6435,0],0.1)
    duff.doit()
