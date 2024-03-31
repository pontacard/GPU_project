import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from one_spin.one_spin_LLG import A_spin
from one_spin.STO_gif import STO_gif
from one_spin.STO_2Dgraph import one_spin_2D
from chaos.STO_Lyapunov import STO_lyapunov

S0 = [0, 0, 1]

t = [0, 500]  # t(時間)が0〜100まで動き、その時のfを求める。
t_eval = np.linspace(*t, 70000)

plotB = [[0, 0, -1.2], [0, 0, 2.4]]

gamma = 0.17
mu_0 = 1.26
mu_h_div_2e = [0.413563, -21]
sta_M = [1.824, 0]  # 飽和磁化(T)で入れる
d = [2, -9]


jdc = [2.2 , 10]
Hdcn = mu_h_div_2e[0] * jdc[0] / (sta_M[0] * d[0])
Hdco = mu_h_div_2e[1] + jdc[1] - (sta_M[1] + d[1])
Hdc = Hdcn * (10 ** Hdco) * 1000 / gamma  # 最後の1000はmTにするため

Kx = 0
Ky = 0
Kz = 1481 - 1448

f = open("/home/tatsumi/Spin/chaos/picture/iniz_Lyapunov.txt","w")


for j in range(3000):
    for omega in [2]:

        ji = 2.3 + j * 0.001
        jac = [ji, 11]
        Hacn = mu_h_div_2e[0] * jac[0] / (sta_M[0] * d[0])
        Haco = mu_h_div_2e[1] + jac[1] - (sta_M[1] + d[1])
        Hac = Hacn * (10 ** Haco) * 1000 / gamma  # 最後の1000はmTにするため

        t = [0, 50]  # t(時間)が0〜100まで動き、その時のfを求める。
        t_eval = np.linspace(*t, 1000000)
        spin_graph = one_spin_2D(0.005, gamma, [0, 0, mu_0 * 159], S0, t, t_eval, [0, 0, 0], [omega, omega, 0], [0, 0, 0], Kx, Ky,
                    mu_0 * Kz, 0, 0, 50, [0, Hac, 0], [0, Hdc, 0], 0.288, 0.537)
        #spin_graph.get_graph()

        t = [0, 10]  # t(時間)が0〜100まで動き、その時のfを求める。
        t_eval = np.linspace(*t, 10000)
        spin_gif = STO_gif(0.005, gamma, [0, 0, mu_0 * 159], S0, t, t_eval, [0, 0, 0], [omega, omega, 0], [0, 0, 0], Kx, Ky,
                    mu_0 * Kz, 0, 0, 10, [0, Hac, 0], [0, Hdc, 0], 0.288, 0.537)
        #spin_gif.make_gif()

        t = [0, 20]  # t(時間)が0〜100まで動き、その時のfを求める。
        t_eval = np.linspace(*t, 200000)
        Lya_expo = 0

        Lyap = STO_lyapunov(0.005, gamma, [0, 0, mu_0 * 159], S0, t, t_eval, 0,omega,[0,0,0],[0, 1, 0], Kx,Ky, mu_0 * Kz, 0, 0, 40000,[0,Hac,0],[0,Hdc,0],0.288,0.537,90000,1000,90,0.1)
        Lya_expo = Lyap.make_trajec()

        #print(f'   {j} Am^-2  {omega} GHz  {Lya_expo}   ')
        if Lya_expo >= 0.1:
            f = open('/home/tatsumi/Spin/chaos/picture/iniz_Lyapunov.txt' , "a")
            f.write(f'{ji}Am^-2 {omega}GHz{Lya_expo}\n')
            f.close()




