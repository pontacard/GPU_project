import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from one_spin.FMR import FMR_spin

def make_poincore(alpha, beta, gamma,B,S0,omega, t,  t_eval,theta,Kx,Ky,Kz,start,stop,sta_B0,end_B0,step_B0):
    B0_ran = [sta_B0, end_B0]
    B0_eval = np.linspace(*B0_ran, step_B0)
    B_list = np.empty(0)
    poi_list = np.empty(0)
    for B0 in B0_eval:
        duf = FMR_spin(alpha,gamma,B,S0,t,t_eval,[0,B0,0],omega,theta,Kx,Ky,Kz,beta,start,stop)
        poi = duf.poincore(5900000,6000000)
        B0_list = [B0] * len(poi)
        poi_list = np.append(poi_list, poi)
        B_list = np.append(B_list,B0_list)
        print(B0)
    plt.scatter(B_list,poi_list, c = 'b',s = 1)
    plt.gca().set_aspect(10)

    plt.savefig("FMR_poincore_3_165_200_aspect.pdf")


S0 = [0.8253, 0.5646, 0]

t = [0, 1000]  # t(時間)が0〜100まで動き、その時のfを求める。
t_eval = np.linspace(*t, 6000000)
mu_0 = 1.2
B0 = 10.5
Bx = 165
omega = 20.232

make_poincore(0.05, 0,0.17,[Bx,0,0],S0,[0,omega,0],t,t_eval,[0,0,0],0 , 200,0,0,3000,7,15,600)
