import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from macro_spin.FMR import FMR_spin
from macro_spin.SOT_sin import Sin_SOT

def FMR_bihuraction(alpha, beta, gamma,ax,B,S0,omega, t,  t_eval,theta,Kx,Ky,Kz,start,stop,sta_B0,end_B0,step_B0):
    B0_ran = [sta_B0, end_B0]
    B0_eval = np.linspace(*B0_ran, step_B0)
    B_list = np.empty(0)
    poi_list = np.empty(0)
    for B0 in B0_eval:
        duf = FMR_spin(alpha,gamma,B,S0,t,t_eval,[0,B0,0],omega,theta,Kx,Ky,Kz,beta,start,stop)
        duf.history()
        poi = duf.poincore(ax,550000,600000)
        B0_list = [B0] * len(poi)
        poi_list = np.append(poi_list, poi)
        B_list = np.append(B_list,B0_list)
    #print(B_list,poi_list)
    plt.ylim(-0.2,1)
    plt.gca().set_aspect(4)
    plt.scatter(B_list,poi_list, c = 'b',s = 1)
    plt.savefig("duffing_poincore_150_176_17.468GHz.pdf")

def SOT_bihuraction(alpha, beta, gamma,ax,B,S0,omega, t,  t_eval,theta,Kx,Ky,Kz,start,stop,sta_B0,end_B0,step_B0):
    B0_ran = [sta_B0, end_B0]
    B0_eval = np.linspace(*B0_ran, step_B0)
    B_list = np.empty(0)
    poi_list = np.empty(0)
    for B0 in B0_eval:
        duf = Sin_SOT(alpha, gamma, B, S0, t, t_eval, [0, B0, 0], omega, theta, Kx, Ky, Kz, beta, start, stop)
        duf.history()
        poi = duf.poincore(ax,450000, 500000)
        B0_list = [B0] * len(poi)
        poi_list = np.append(poi_list, poi)
        B_list = np.append(B_list, B0_list)
    # print(B_list,poi_list)
    plt.ylim(-0.2, 1)
    plt.gca().set_aspect(8)
    plt.scatter(B_list, poi_list, c='b', s=1)
    plt.savefig("SOT_bihuraction_160_200_20.2244GHz.pdf")


S0 = [4/5, 3/5, 0]

t = [0, 500]  # t(時間)が0〜100まで動き、その時のfを求める。
t_eval = np.linspace(*t, 500001)
mu_0 = 1.2
B0 = 10.5
Bx = 160
omega = 20.2244

#FMR_bihuraction(0.05, 0,0.17,1,[Bx,0,0],S0,[0,omega,0],t,t_eval,[0,0,0],0 , 200,0,0,900,3,5,6)
SOT_bihuraction(0.05, 0,0.17,1,[Bx,0,0],S0,[0,omega,0],t,t_eval,[0,0,0],0 , 200,0,0,9000,10,60,501)
