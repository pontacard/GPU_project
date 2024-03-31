import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from Spin_Lyapunov import Lyapunov
from one_spin.FMR import FMR_spin
from one_spin.FMR_gif import FMR_gif


class FMR_lyapunov(Lyapunov):
    def __init__(self,alpha,gamma,B,S0,t,t_eval,Amp,omega,theta,Kx,Ky,Kz,beta,spin_start,spin_stop,lya_start_step,lya_cycle,delta_t1):
        super().__init__(alpha,gamma,B,S0,t,t_eval,Amp,omega,theta,Kx,Ky,Kz,beta,spin_start,spin_stop, lya_start_step, lya_cycle, delta_t1)


    def make_trajec(self):
        delta_theta = self.omega * self.delta_t1
        per_theta = np.array(self.theta) + delta_theta
        perturb_spin = {}


        spin = FMR_spin(self.alpha, self.gamma, self.B, self.S0, self.t, self.t_eval, self.Amp, self.omega, self.theta, self.Kx, self.Ky, self.Kz, self.beta, self.spin_start, self.spin_stop)
        perturb_spin[0] = FMR_spin(self.alpha, self.gamma, self.B, self.S0, self.t, self.t_eval, self.Amp, self.omega,
                                  per_theta, self.Kx, self.Ky, self.Kz, self.beta, self.spin_start, self.spin_stop)
        plotB = [[0, 0, -1.2], [0, 0, 2.4]]

        spin_gif = FMR_gif(self.alpha, self.gamma, self.B, self.S0, self.t, self.t_eval, self.Amp, self.omega,
                                  per_theta, self.Kx, self.Ky, self.Kz, self.beta, self.spin_start, self.spin_stop, plotB)
        spin_gif.make_gif()

        spin.history()
        perturb_spin[0].history()
        spin0_log = spin.S.T
        per_spin0_log = perturb_spin[0].S.T

        S_t0 = spin0_log[self.lya_start]            #リヤプノフ指数を図るときの最初のt(最初から(t=0)距離を測り始めてしまうと最初の距離が0になってしまうため。)
        perS_t0 = per_spin0_log[self.lya_start]

        print(spin0_log,per_spin0_log)

        #print(self.alpha, self.gamma, self.B, self.S0, self.t, self.t_eval, self.Amp, self.omega, self.theta, self.Kx, self.Ky, self.Kz, self.beta, self.spin_start, self.spin_stop)
        print(per_theta)

        distance_t0 = self.distance(S_t0, perS_t0,self.delta_t1)
        a = 1/distance_t0
        print("D",distance_t0)
        print(np.log(a))
        S_t0_delt = spin0_log[self.lya_start + 1]
        perS_t0_delt = per_spin0_log[self.lya_start + 1]
        p1 = (self.distance(S_t0_delt, perS_t0_delt, self.delta_t1))

        lya_expo = [p1]
        dt = self.delta_t1[1]
        ex_spin_log = spin0_log
        sum_log_ly = np.log(p1)



        for k in range(1 , self.lya_cycle):
            #print(k)
            #print(lya_expo[k-1])
            n_omega = np.linalg.norm(self.omega)
            dt = dt / (lya_expo[k-1]  * a * np.linalg.norm(self.omega))
            #print(n_omega)
            d_theta = dt
            ptheta = np.array(self.theta) + d_theta
            print(ptheta)
            perturb_spin[k] = FMR_spin(self.alpha, self.gamma, self.B, self.S0, self.t, self.t_eval, self.Amp, self.omega ,ptheta, self.Kx, self.Ky, self.Kz, self.beta, self.spin_start, self.spin_stop)
            perturb_spin[k].history()
            spin_log = perturb_spin[k].S.T

            pk = self.distance(spin0_log[self.lya_start + k], spin_log[self.lya_start + k],dt)
            #print("pk",pk)
            lya_expo.append(pk)
            #print(lya_expo)
            print(pk,np.log(pk))

            sum_log_ly += np.log(pk)


        step_width = (self.t[1] - self.t[0]) / len(self.t_eval)


        Lyapnov_exponent = sum_log_ly / (self.lya_cycle) + np.log(a)

        print("here",Lyapnov_exponent)

if __name__ == '__main__':
    S0 = [np.sqrt(0.001), 0, np.sqrt(0.999)]

    t = [0, 50]  # t(時間)が0〜100まで動き、その時のfを求める。
    t_eval = np.linspace(*t, 12000)

    plotB = [[0, 0, -1.2], [0, 0, 2.4]]

    gamma = 0.17
    mu_h_div_2e = [0.824, -21]
    sta_M = [1.4, 0]  # 飽和磁化(T)で入れる
    theta = [-2, -1]
    j = [2.5, 10]
    d = [1, -9]
    Hsn = mu_h_div_2e[0] * theta[0] * j[0] / (sta_M[0] * d[0])
    Hso = mu_h_div_2e[1] + theta[1] + j[1] - (sta_M[1] + d[1])
    Hs = Hsn * (10 ** Hso) * 1000 / gamma  # 最後の1000はmTにするため
    print(Hs)

    Lyap = FMR_lyapunov(0.005, gamma, [0, 0, 200], S0, t, t_eval, [1200, 1200, 0],[0.3,0.3,0],[0,0,0], 0, 0, 2000, 0, 0, 50, 3000, 40, [0.01,0.01,0])
    Lyap.make_trajec()
