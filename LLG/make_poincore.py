import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from chaos.duffing import Duffing

def make_poincore(alpha, beta, gamma,omega, t,  t_eval, X0,sta_B,end_B,step_B):
    B_ran = [sta_B, end_B]
    B_eval = np.linspace(*B_ran, step_B)
    B_list = np.empty(0)
    poi_list = np.empty(0)
    for B in B_eval:
        duf = Duffing(alpha, beta, gamma, B,omega, t,  t_eval, X0)
        poi = duf.poincore(19900000,20000000)
        B0_list = [B] * len(poi)
        poi_list = np.append(poi_list, poi)
        B_list = np.append(B_list,B0_list)
    #print(B_list,poi_list)
    plt.scatter(B_list,poi_list, c = 'b',s = 1)
    plt.gca().set_aspect(5)
    plt.rcParams["font.size"] = 20
    plt.savefig(f"duffing_poincore_alpha_{alpha}_{beta}_{gamma}_{omega}GHz.pdf")


t = [0,2000]
t_eval = np.linspace(*t, 20000001)
make_poincore(2,130,176,16.08,t,t_eval,[0.859,0,0], 30, 60, 401)
