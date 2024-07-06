import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from macro_spin.FMR import FMR_spin
from macro_spin.SOT_sin import Sin_SOT
from macro_spin.thermal import Thermal_spin
import csv

def FMR_bihuraction(alpha, beta, gamma,ax,B,S0,omega, t,  t_eval,theta,Kx,Ky,Kz,start,stop,sta_B0,end_B0,step_B0):
    B0_ran = [sta_B0, end_B0]
    B0_eval = np.linspace(*B0_ran, step_B0)
    B_list = np.empty(0)
    poi_list = np.empty(0)
    for B0 in B0_eval:
        duf = FMR_spin(alpha,gamma,B,S0,t,t_eval,[0,B0,0],omega,theta,Kx,Ky,Kz,beta,start,stop)
        duf.history()
        poi = duf.poincore(ax,7990000,8000001)
        B0_list = [B0] * len(poi)
        poi_list = np.append(poi_list, poi)
        B_list = np.append(B_list,B0_list)
    #print(B_list,poi_list)
    #plt.ylim(-0.2,1)
    #plt.gca().set_aspect(4)

    #with open(f"FMR_duffing_poincore_{B[0]}_{B[1]}_{B[2]}_{Kx}_{Ky}_{Kz}_{omega[1]}_paper.csv","w") as f:
        #writer = csv.writer(f)
        #writer.writerow(np.stack([B_list,poi_list]))

    np.savetxt(f"csv/FMR_duffing_poincore_{B[0]}_{B[1]}_{B[2]}_{Kx}_{Ky}_{Kz}_{omega[1]}_35.5-36_paper.txt", np.stack([B_list,poi_list]))

    #plt.scatter(B_list,poi_list, c = 'b',s = 1)
    #plt.yticks([-1, 0, 1])
    #plt.savefig(f"FMR_duffing_poincore_{B[0]}_{B[1]}_{B[2]}_{Kx}_{Ky}_{Kz}_{omega[1]}_paper.pdf")
    #plt.show()

def SOT_bihuraction(alpha, beta, gamma,ax,B,S0,omega, t,  t_eval,theta,Kx,Ky,Kz,start,stop,sta_B0,end_B0,step_B0):
    B0_ran = [sta_B0, end_B0]
    B0_eval = np.linspace(*B0_ran, step_B0)
    B_list = np.empty(0)
    poi_list = np.empty(0)
    for B0 in B0_eval:
        duf = Sin_SOT(alpha, gamma, B, S0, t, t_eval, [0, B0, 0], omega, theta, Kx, Ky, Kz, beta, start, stop)
        duf.history()
        poi = duf.poincore(ax,7880000,8000001)
        B0_list = [B0] * len(poi)
        poi_list = np.append(poi_list, poi)
        B_list = np.append(B_list, B0_list)
    # print(B_list,poi_list)
    #plt.ylim(-0.2, 1)
    np.savetxt(f"csv/SOT_duffing_poincore_{B[0]}_{B[1]}_{B[2]}_{Kx}_{Ky}_{Kz}_{omega[1]}_paper.txt", np.stack([B_list,poi_list]))
    #plt.gca().set_aspect(4)
    #plt.scatter(B_list, poi_list, c='b', s=1)
    #plt.savefig(f"SOT_Sx_bihuraction_[{B[0]}_{B[1]}_{B[2]}]_[{Kx}_{Ky}_{Kz}]_{omega[1]}_komine.pdf")


def FMR_thermal_bihuraction(alpha, beta, gamma,ax,B,S0,omega, t,  t_eval,theta,Kx,Ky,Kz,start,stop,sigma,ther_dt,sta_B0,end_B0,step_B0):
    B0_ran = [sta_B0, end_B0]
    B0_eval = np.linspace(*B0_ran, step_B0)
    B_list = np.empty(0)
    poi_list = np.empty(0)
    for B0 in B0_eval:
        duf = Thermal_spin(alpha, gamma, B, S0, t, t_eval, [0, B0, 0], omega, theta, Kx, Ky, Kz, beta,sigma,ther_dt ,start, stop)
        duf.history()
        poi = duf.poincore(ax,4995000, 5000001)
        B0_list = [B0] * len(poi)
        poi_list = np.append(poi_list, poi)
        B_list = np.append(B_list, B0_list)
    # print(B_list,poi_list)
    #plt.ylim(-0.2, 1)
    plt.gca().set_aspect(6)
    plt.scatter(B_list, poi_list, c='b', s=1)
    plt.savefig(f"FMR_thermal_poincore_{B[0]}_{B[1]}_{B[2]}_{Kx}_{Ky}_{Kz}_{omega[1]}GHz_{sigma}.pdf")



ther_dt = 0.01


alpha = 0.05
gamma = 0.17

kb = [1.38, -23]
sta_M = [1.4, 0]  # 飽和磁化(T)で入れる
T = [3.0, 3]
V = [np.pi * 60 * 6 * 1, -27]
ns = [1, -9]
gamma_use = [1.76/(2 *np.pi) , 11]
mu_0 = [1.256, -6]
Bthe_n = kb[0] * T[0] * mu_0[0] / (sta_M[0] * V[0] * gamma_use[0] * ns[0])
Bthe_o = kb[1] + T[1] + mu_0[1] - (sta_M[1] + V[1] + gamma_use[1] + ns[1])
Bthe_2 = 2 * alpha * Bthe_n * (10 ** Bthe_o)
sigma_Bthe = np.sqrt(Bthe_2) * 1000 * ther_dt# 最後の1000はmTにするため
print(sigma_Bthe)


S0 = [0.825, 0.565, 0]

t = [0, 4000]  # t(時間)が0〜100まで動き、その時のfを求める。
t_eval = np.linspace(*t, 8000001)
mu_0 = 1.2
B0 = 10.5
Bx = 160
omega = 28.65

#FMR_thermal_bihuraction(0.05, 0,0.17,1,[Bx,0,0],S0,[0,omega,0],t,t_eval,[0,0,0],0 , 200,0,0,9000,sigma_Bthe,ther_dt,4,25,301)
FMR_bihuraction(0.05, 0,0.176335977,1,[Bx,0,0],S0,[0,omega,0],t,t_eval,[0,0,0],0,0,0,0,9000,35.5,36,21)
#SOT_bihuraction(0.05, 0,0.176335977,1,[Bx,0,0],S0,[0,omega,0],t,t_eval,[0,0,0],0 , 200,0,0,9000,4,25,211)
