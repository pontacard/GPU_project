import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from chaos.New_Spin_Lyapunov import Lyapunov
from one_spin.STO import STO_spin


class STO_lyapunov(Lyapunov):
    def __init__(self,alpha,gamma,B,S0,t,t_eval,Amp,omega,theta ,unit ,Kx,Ky,Kz,beta,spin_start,spin_stop,STO_ac_effH,STO_dc_effH,lambdaa,eta,lya_start_step,lya_step_width,lya_cycle,delta_t1):
        super().__init__(alpha,gamma,B,S0,t,t_eval,Amp,omega,theta,unit ,Kx,Ky,Kz,beta,spin_start,spin_stop, lya_start_step,lya_step_width, lya_cycle, delta_t1)
        self.STO_ac_effH = STO_ac_effH
        self.STO_dc_effH = STO_dc_effH
        self.lambdaa = lambdaa
        self.eta = eta


    def make_trajec(self):
        dt = (self.t[1] - self.t[0]) / len(self.t_eval)
        Amp_v = self.Amp * self.unit
        omega_v = self.omega * self.unit
        theta_v = self.theta * self.unit

        per_theta = (self.theta + self.delta_t1) * self.unit
        #print(per_theta)
        perturb_spin = {}

        spin = STO_spin(self.alpha, self.gamma, self.B, self.S0, self.t, self.t_eval, Amp_v, omega_v, theta_v, self.Kx,
                       self.Ky, self.Kz, self.beta, self.spin_start, self.spin_stop,self.STO_ac_effH,self.STO_dc_effH,self.lambdaa,self.eta)
        perturb_spin[0] = STO_spin(self.alpha, self.gamma, self.B, self.S0, self.t, self.t_eval, Amp_v, omega_v,
                                  per_theta, self.Kx, self.Ky, self.Kz, self.beta, self.spin_start, self.spin_stop,self.STO_ac_effH,self.STO_dc_effH,self.lambdaa,self.eta)


        spin.history()
        perturb_spin[0].history()
        spin0_log = spin.S.T
        per_spin0_log = perturb_spin[0].S.T

        S_t0 = spin0_log[self.lya_start]            #リヤプノフ指数を図るときの最初のt(最初から(t=0)距離を測り始めてしまうと最初の距離が0になってしまうため。)
        perS_t0 = per_spin0_log[self.lya_start]

        lya_expo = []
        sum_log_ly = 0

        #print(self.alpha, self.gamma, self.B, self.S0, self.t, self.t_eval, self.Amp, self.omega, self.theta, self.Kx, self.Ky, self.Kz, self.beta, self.spin_start, self.spin_stop)
        #print(per_theta)

        distance_t0 = self.distance(S_t0, perS_t0, self.delta_t1)
        epsi = distance_t0
        d_theta = self.delta_t1
        print(self.l(S_t0, perS_t0))
        m_kdelt = self.S0
        ptheta = per_theta


        for k in range(1,self.lya_cycle):
            perturb_spin = STO_spin(self.alpha, self.gamma, self.B, m_kdelt, [0, 10 * self.lya_step_width * dt],
                                       np.linspace(*[0, 10 * self.lya_step_width * dt], 10 * self.lya_step_width), Amp_v,
                                       omega_v,
                                       ptheta, self.Kx, self.Ky, self.Kz, self.beta, self.spin_start, self.spin_stop,
                                       self.STO_ac_effH, self.STO_dc_effH, self.lambdaa, self.eta)

            perturb_spin.history()
            per_spin_log = perturb_spin.S.T

            S0_delt = spin0_log[self.lya_start + k * self.lya_step_width]
            perS0_delt = per_spin_log[self.lya_step_width]
            pk = self.distance(S0_delt, perS0_delt, self.delta_t1) / epsi

            lya_expo.append(pk)
            sum_log_ly += np.log(pk)

            mcro = np.cross(S0_delt, perS0_delt)
            # print(spin0_log[self.lya_start + (k + 1) * self.lya_step_width], spin_log[self.lya_step_width])
            nmcro = mcro / np.linalg.norm(mcro)
            # print(nmcro)
            l_kdash = self.l(S0_delt, perS0_delt)
            l_k = l_kdash / pk
            cos = np.cos(l_k)
            sin = np.sin(l_k)
            R = np.array([[(nmcro[0] ** 2) * (1 - cos) + cos, (nmcro[0] * nmcro[1]) * (1 - cos) - nmcro[2] * sin,
                           (nmcro[0] * nmcro[2]) * (1 - cos) + nmcro[1] * sin],
                          [(nmcro[0] * nmcro[1]) * (1 - cos) + nmcro[2] * sin, (nmcro[1] ** 2) * (1 - cos) + cos,
                           (nmcro[1] * nmcro[2]) * (1 - cos) - nmcro[0] * sin],
                          [(nmcro[0] * nmcro[2]) * (1 - cos) - nmcro[1] * sin,
                           (nmcro[1] * nmcro[2]) * (1 - cos) + nmcro[0] * sin, (nmcro[2] ** 2) * (1 - cos) + cos]])
            m_kdelt = np.dot(R, S0_delt)

            print("0になるはず", epsi , self.distance(S0_delt, m_kdelt, d_theta/ pk))

            d_theta = d_theta / pk
            print(d_theta)
            k_start = dt * (self.lya_start + k * self.lya_step_width)
            k_start_theta = self.omega * k_start
            ptheta = (self.theta + d_theta + k_start_theta) * self.unit

            #print("lk",l_k)

        #print(sum_log_ly)
        #print(dt)

        Lyapnov_exponent = sum_log_ly / (self.lya_cycle * dt * self.lya_step_width)


        print("here",Lyapnov_exponent)
        return Lyapnov_exponent

if __name__ == '__main__':
    S0 = [1, 0, 0]

    t = [0, 20]  # t(時間)が0〜100まで動き、その時のfを求める。
    t_eval = np.linspace(*t, 2000)

    plotB = [[0, 0, -1.2], [0, 0, 2.4]]

    gamma = 0.17
    mu_0 = 1.26
    mu_h_div_2e = [0.413563, -21]
    sta_M = [1.824, 0]  # 飽和磁化(T)で入れる
    jac = [3, 11]
    d = [2, -9]
    Hacn = mu_h_div_2e[0] * jac[0] / (sta_M[0] * d[0])
    Haco = mu_h_div_2e[1] + jac[1] - (sta_M[1] + d[1])
    Hac = Hacn * (10 ** Haco) * 1000 / gamma  # 最後の1000はmTにするため
    #print(Hac)

    jdc = [2.2, 10]
    Hdcn = mu_h_div_2e[0] * jdc[0] / (sta_M[0] * d[0])
    Hdco = mu_h_div_2e[1] + jdc[1] - (sta_M[1] + d[1])
    Hdc = Hdcn * (10 ** Hdco) * 1000 / gamma  # 最後の1000はmTにするため

    Kx = 0
    Ky = 0
    Kz = 1481 - 1448

    Lyap = STO_lyapunov(0.005, gamma, [0, 0, mu_0 * 159], S0, t, t_eval, 0,1.8,[0,0,0],[0, 1, 0], Kx,Ky, mu_0 * Kz, 0, 0, 40000,[0,Hac,0],[0,Hdc,0],0.288,0.537,1000,100,8,0.1)
    Lyap.make_trajec()

