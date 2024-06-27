import csv

import numpy as np
import matplotlib.pyplot as plt
from chaos.duffing import Duffing
from chaos.FMR_duffing import FMR_Duffing
from chaos.SOT_duffing import SOT_Duffing
from chaos.FMR import FMR
from chaos.thermal_FMR import Thermal_FMR

def Lyapunov_map(alpha, beta, gamma,omega, t,  t_eval, X0,sta_B,end_B,step_B):
    B_ran = [sta_B, end_B]
    B_eval = np.linspace(*B_ran, step_B)
    B_list = np.empty(0)
    poi_list = np.empty(0)
    for B in B_eval:
        duf = Duffing(alpha, beta, gamma, B, omega, t, t_eval, X0)
        Lya = duf.matsunaga_Lyapunov([0.001,0,0],200,0.02)
        print(Lya)
        poi_list = np.append(poi_list, [Lya])
        B_list = np.append(B_list, [B])
    # print(B_list,poi_list)
    with open("csv/duffing_Lyapunovmap_{beta}_{gamma}_{omega}GHz.csv") as f:
        writer = csv.writer(f)
        writer.writerow(np.stack([B_list,poi_list]))
    plt.scatter(B_list, poi_list, c='b', s=1)
    plt.gca().set_aspect(8)
    plt.savefig(f"duffing_Lyapunovmap_{beta}_{gamma}_{omega}GHz.pdf")

def FMR_Lyapunov_map(alpha, gamma,Bx,Ky,omega, t,  t_eval, S0,sta_B,end_B,step_B,per,Lya_step,start_step,aspect = 12):
    B_ran = [sta_B, end_B]
    B_eval = np.linspace(*B_ran, step_B)
    B_list = np.empty(0)
    Lya_list = np.empty(0)
    for B in B_eval:
        duf = FMR_Duffing(alpha, gamma,Bx,Ky, B, omega, t, t_eval, S0)
        Lya = duf.matsunaga_Lyapunov(per, Lya_step,5,start_step)
        #print(Lya)
        Lya_list = np.append(Lya_list, [Lya])
        B_list = np.append(B_list, [B])
    # print(B_list,poi_list)
    np.savetxt(f"csv/FMR_Lyapunovmap_Bx_{Bx}_Ky_{Ky}_{omega}GHz._start_step_{start_step}_Lyastep_{Lya_step}_paper.txt", np.stack([B_list,Lya_list]))

    #plt.scatter(B_list, Lya_list, c='b', s=5)
    #plt.yticks([-2,-1,0])
    #plt.gca().set_aspect(aspect)
    #plt.savefig(f"FMR_Lyapunovmap_Bx_{Bx}_Ky_{Ky}_{omega}GHz._start_step_{start_step}_Lyastep_{Lya_step}_for_paper.pdf")
    #plt.show()

def SOT_Lyapunov_map(alpha, gamma,Bx,Ky,omega, t,  t_eval, S0,sta_SOT,end_SOT,step_SOT,per,Lya_step,start_step,aspect = 8):
    SOT_ran = [sta_SOT, end_SOT]
    SOT_eval = np.linspace(*SOT_ran, step_SOT)
    SOT_list = np.empty(0)
    Lya_list = np.empty(0)
    for SOT in SOT_eval:
        duf = SOT_Duffing(alpha, gamma,Bx,Ky, SOT, omega, t, t_eval, S0)
        Lya = duf.matsunaga_Lyapunov(per, Lya_step,5,start_step)
        #print(Lya)
        Lya_list = np.append(Lya_list, [Lya])
        SOT_list = np.append(SOT_list, [SOT])
    # print(B_list,poi_list)
    np.savetxt(f"csv/SOT_Lyapunovmap_Bx_{Bx}_Ky_{Ky}_{omega}GHz._start_step_{start_step}_Lyastep_{Lya_step}_paper.txt",
               np.stack([SOT_list, Lya_list]))

    # plt.scatter(B_list, Lya_list, c='b', s=5)
    # plt.yticks([-2,-1,0])
    # plt.gca().set_aspect(aspect)
    # plt.savefig(f"FMR_Lyapunovmap_Bx_{Bx}_Ky_{Ky}_{omega}GHz._start_step_{start_step}_Lyastep_{Lya_step}_for_paper.pdf")
    # plt.show()
def ax_FMR_Lyapunov_map(alpha, gamma,B,K,ax,omega,phase, t,  t_eval, S0,sta_B,end_B,step_B,per,Lya_step,start_step,aspect = 8):
    B_ran = [sta_B, end_B]
    B_eval = np.linspace(*B_ran, step_B)
    B_list = np.empty(0)
    Lya_list = np.empty(0)
    Bac = np.zeros(3)
    for Bacsc in B_eval:
        Bac[ax] = Bacsc
        duf = FMR(alpha, gamma,B,K, Bac, omega,phase, t, t_eval, S0)
        Lya = duf.matsunaga_Lyapunov(per, Lya_step,5,start_step)
        #print(Lya)
        Lya_list = np.append(Lya_list, [Lya])
        B_list = np.append(B_list, [Bacsc])
    #print(B_list, Lya_list)
    plt.scatter(B_list, Lya_list, c='b', s=1)
    plt.gca().set_aspect(aspect)
    plt.savefig(f"FMR_Lyapunovmap_Bx={B}_Ky={K}_{omega}GHz.pdf")

def thermal_FMR_Lyapunov_map(alpha, gamma,B,K,ax,omega,phase,sigma_Bthe, ther_dt, t,  t_eval, S0,sta_B,end_B,step_B,aspect = 8):
    B_ran = [sta_B, end_B]
    B_eval = np.linspace(*B_ran, step_B)
    B_list = np.empty(0)
    Lya_list = np.empty(0)
    Bac = np.zeros(3)
    sigma_Bthe = sigma_Bthe
    ther_dt = ther_dt
    for Bacsc in B_eval:
        Bac[ax] = Bacsc
        duf = Thermal_FMR(alpha, gamma, B,K, Bac,omega, phase, sigma_Bthe, ther_dt, t, t_eval, S0)
        Lya = duf.matsunaga_Lyapunov([0.01,0,0], 200, 0.03)
        Lya_list = np.append(Lya_list, [Lya])
        B_list = np.append(B_list, [Bacsc])
    # print(B_list,poi_list)
    # plt.ylim(-0.2, 1)
    plt.gca().set_aspect(aspect)
    plt.scatter(B_list, Lya_list, c='b', s=1)
    plt.savefig(f"Lyapunovmap/FMR_thermal_Lyapunov_map_{B[0]}_{B[1]}_{B[2]}_{K[0]}_{K[1]}_{K[2]}_{omega}GHz_{sigma_Bthe}.pdf")


t = [0,800]
t_eval = np.linspace(*t, 8000001)
#Lyapunov_map(1,32,176,8.092,t,t_eval,[0.4264,0,0], 2, 10, 400)
FMR_Lyapunov_map(0.05,0.176335977,160,0,20.232,t,t_eval,[np.pi/2,0.6005,0],4,25,421,[0.01,0,0], 2001,6000000,aspect = 3.5)
#SOT_Lyapunov_map(0.05,0.176335977,165,200,20.232,t,t_eval,[np.pi/2,0.6435,0],0,250,501,[0.01,0,0], 10001,7000000,aspect = 2)

B = [0,160,0]
K = [0,200,0]
phase = [0,0,0]
#ax_FMR_Lyapunov_map(0.05,0.17,B,K,1,51.019,phase,t,t_eval,[np.pi/2,0.6435,0],4,40,2)
#thermal_FMR_Lyapunov_map(0.05,0.17,B,K,1,20.232,phase,0.1,0.01,t,t_eval,[np.pi/2,0.6435,0],4,20,161)1

