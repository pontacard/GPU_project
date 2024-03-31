import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from chaos.duffing import Duffing

def duffing_bihuraction(alpha, beta, gamma,omega, t,  t_eval, X0,sta_B,end_B,step_B):
    B_ran = [sta_B, end_B]
    B_eval = np.linspace(*B_ran, step_B)
    for B in B_eval:
        duf = Duffing(alpha, beta, gamma, B,omega, t,  t_eval, X0)
        poi = duf.poincore(170000,200000)
        B_list = [B] * len(poi)
        plt.scatter(B_list,poi, c = 'b',s = 5)
        plt.gca().set_aspect(5)

    plt.show()





t = [0,50]
t_eval = np.linspace(*t, 200000)
duffing_bihuraction(1,130,176,3.5,t,t_eval,[0.42,0.0001,3.5], 20, 50, 4)