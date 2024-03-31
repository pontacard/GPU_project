import numpy as np
import matplotlib.pyplot as plt
from chaos.duffing import Duffing
from chaos.FMR_duffing import FMR_Duffing
from chaos.SOT_duffing import SOT_Duffing
from chaos.FMR import FMR

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
    plt.scatter(B_list, poi_list, c='b', s=1)
    plt.gca().set_aspect(8)
    plt.savefig(f"duffing_Lyapunovmap_{beta}_{gamma}_{omega}GHz.pdf")

def FMR_Lyapunov_map(alpha, gamma,Bx,Ky,omega, t,  t_eval, S0,sta_B,end_B,step_B,aspect = 8):
    B_ran = [sta_B, end_B]
    B_eval = np.linspace(*B_ran, step_B)
    B_list = np.empty(0)
    Lya_list = np.empty(0)
    for B in B_eval:
        duf = FMR_Duffing(alpha, gamma,Bx,Ky, B, omega, t, t_eval, S0)
        Lya = duf.matsunaga_Lyapunov([0.01, 0, 0], 200,0.02)
        #print(Lya)
        Lya_list = np.append(Lya_list, [Lya])
        B_list = np.append(B_list, [B])
    # print(B_list,poi_list)
    plt.scatter(B_list, Lya_list, c='b', s=1)
    plt.gca().set_aspect(aspect)
    plt.savefig(f"FMR_Lyapunovmap_Bx={Bx}_Ky={Ky}_{omega}GHz._light.pdf")

def SOT_Lyapunov_map(alpha, gamma,Bx,Ky,omega, t,  t_eval, S0,sta_SOT,end_SOT,step_SOT,aspect = 8):
    SOT_ran = [sta_SOT, end_SOT]
    SOT_eval = np.linspace(*SOT_ran, step_SOT)
    SOT_list = np.empty(0)
    Lya_list = np.empty(0)
    for SOT in SOT_eval:
        duf = SOT_Duffing(alpha, gamma,Bx,Ky, SOT, omega, t, t_eval, S0)
        Lya = duf.matsunaga_Lyapunov([0.01, 0, 0], 200,0.02)
        #print(Lya)
        Lya_list = np.append(Lya_list, [Lya])
        SOT_list = np.append(SOT_list, [SOT])
    # print(B_list,poi_list)
    plt.scatter(SOT_list, Lya_list, c='b', s=1)
    plt.gca().set_aspect(aspect)
    plt.savefig(f"SOT_Lyapunovmap_Bx={Bx}_Ky={Ky}_{omega}GHz_light.pdf")

def ax_FMR_Lyapunov_map(alpha, gamma,B,K,ax,omega,phase, t,  t_eval, S0,sta_B,end_B,step_B,aspect = 8):
    B_ran = [sta_B, end_B]
    B_eval = np.linspace(*B_ran, step_B)
    B_list = np.empty(0)
    Lya_list = np.empty(0)
    Bac = np.zeros(3)
    for Bacsc in B_eval:
        Bac[ax] = Bacsc
        duf = FMR(alpha, gamma,B,K, Bac, omega,phase, t, t_eval, S0)
        Lya = duf.matsunaga_Lyapunov([0.01, 0, 0], 200,0.03)
        print(Lya)
        Lya_list = np.append(Lya_list, [Lya])
        B_list = np.append(B_list, [Bacsc])
    print(B_list, Lya_list)
    plt.scatter(B_list, Lya_list, c='b', s=1)
    plt.gca().set_aspect(aspect)
    plt.savefig(f"FMR_Lyapunovmap_Bx={B}_Ky={K}_{omega}GHz.pdf")


t = [0,800]
t_eval = np.linspace(*t, 800001)
#Lyapunov_map(1,32,176,8.092,t,t_eval,[0.4264,0,0], 2, 10, 400)
#FMR_Lyapunov_map(0.05,0.17,165,200,20.232,t,t_eval,[np.pi/2,0.6435,0],4,12,200,aspect = 2)
#SOT_Lyapunov_map(0.05,0.17,165,200,20.232,t,t_eval,[np.pi/2,0.6435,0],10,24,400,aspect = 3)

B = [165,0,0]
K = [0,200,-1000]
phase = [0,0,0]
ax_FMR_Lyapunov_map(0.05,0.17,B,K,1,51.019,phase,t,t_eval,[np.pi/2,0.6435,0],4,40,200)

