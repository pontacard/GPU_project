import scipy as sc
import numpy as np
import matplotlib.pyplot as plt

def distance( S0, S1, del_t):
    S0 = np.array(S0)
    S1 = np.array(S1)
    print(S0, S1)
    # print(np.dot(S0,S1))

    Sdot = np.dot(S0, S1)
    Sdot = np.around(Sdot, 12)
    l_cos_dis = np.arccos(Sdot)
    l_cross_dis = (1 - Sdot) / 2
    # print(l_cos_dis)
    # print( (np.linalg.norm(self.omega) * np.linalg.norm(self.delta_t1))**2)
    # print(del_t)
    print(l_cos_dis, del_t)
    dist = np.sqrt(l_cos_dis ** 2 + del_t ** 2)
    return dist

print(distance([0.50971447, 0.02435797, 0.85999875], [ 0.49967767, -0.04212856,  0.86518634],1))