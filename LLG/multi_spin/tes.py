import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation

n = 100
count = 0

def S(S,t):  # 関数を設定
    global count
    dSdt = np.empty(0)
    B = [0, 0, 5]
    ba, fo = 0,0
    count += 1



    for i in range(n):
        Si = [S[2* (3 * i + 0)], S[2*(3 * i + 1)], S[2 * (3 * i + 2)]]
        Sib = []
        Sif = []
        if i == 0:
            fo = i + 1
            Sib = [0,0,0]
            Sif = [S[2*(3 * fo + 0)], S[2*(3 * fo + 1)], S[2*(3 * fo + 2)]]

        elif i == n-1:
            ba = i - 1
            Sib = [S[2*(3 * ba + 0)], S[2*(3 * ba + 1)], S[2*(3 * ba + 2)]]
            Sif = [0,0,0]

        else:
            fo = i + 1
            ba = i - 1
            Sib = [S[2*(3 * ba + 0)], S[2*(3 * ba + 1)], S[2*(3 * ba + 2)]]
            Sif = [S[2*(3 * fo + 0)], S[2*(3 * fo + 1)], S[2*(3 * fo + 2)]]

        Snorm = np.linalg.norm(Si)

        #print(S[2*(3 * i + 0) + 1])

        dSixdt = 2 * (Si[1] * (Sib[2] + Sif[2])  - Si[2] * (Sib[1] + Sif[1] + -S[2*(3 * i + 1) + 1])) - ((0000.1/Snorm) * (Si[1] * (Si[0] * (Sib[1] + Sif[1])  - Si[1] * (Sib[0] + Sif[0])) - Si[2]* (Si[2] * (Sib[0] + Sif[0])  - Si[0] * (Sib[2] + Sif[2]))) )
        dSiydt = 2 * (Si[2] * (Sib[0] + Sif[0] + S[2*(3 * i + 0) + 1] )  - Si[0] * (Sib[2] + Sif[2])) - ((0000.1/Snorm) * (Si[2] * (Si[1] * (Sib[2] + Sif[2])  - Si[2] * (Sib[1] + Sif[1])) - Si[0]* (Si[0] * (Sib[1] + Sif[1])  - Si[1] * (Sib[0] + Sif[0]))) )
        dSizdt = 2 * (Si[0] * (Sib[1] + Sif[1] + -S[2*(3 * i + 1) + 1])  - Si[1] * (Sib[0] + Sif[0] + S[2*(3 * i + 0) + 1])) - ((0000.1/Snorm) * (Si[0] * (Si[2] * (Sib[0] + Sif[0])  - Si[0] * (Sib[2] + Sif[2])) - Si[1]* (Si[1] * (Sib[2] + Sif[2])  - Si[2] * (Sib[1] + Sif[1]))) )
        aa =0

        if S[2*(3 * i + 1) + 1]:
            dSdt = np.append(dSdt,[dSixdt,0, dSiydt,-S[2*(3 * i + 1) + 1]*0.01 , dSizdt,0])
            continue
        else:
            aa=1
        if int(S[2*(3 * i + 0) + 1]):
            dSdt = np.append(dSdt, [dSixdt, -S[2*(3 * i + 0) + 1]*0.1, dSiydt, 0, dSizdt, 0])
            continue
        elif aa:
            dSdt = np.append(dSdt, [dSixdt, 0, dSiydt, 0, dSizdt, 0])

    #print(dSdt)

    dSdt = np.reshape(dSdt,(1,-1))

    return dSdt[0]

S0 = np.zeros((n,3,2))
S0[0][2][0] = 1
S0[0][2][1] = 0
S0[0][1][1] = 100

for i in range(n):
    if i == 0:
        continue
    S0[i][2][0] = 1
#S0 = np.array([[1,0,0],[0,0,1],[0,0,1]])
#print(S0)
S0 = np.reshape(S0,(1,-1))
#print(S0)


t = np.linspace(0, 100, 1000)   # t(時間)が0〜3まで動き、その時のfを求める。
#print(t)

y = sc.integrate.odeint(S, S0[0], t)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

frames = []

S = np.reshape(y,(-1,n,3,2))

#print(S)

def get_spin_vec(t,i):
    Ox,Oy,Oz = 3*i,0,0
    x = S[t][i][0][0]
    y = S[t][i][1][0]
    z = S[t][i][2][0]
    return Ox, Oy, Oz, x, y, z

quiver_dic = dict()
for i in range(n):
    quiver_dic[i] = ax.quiver(*get_spin_vec(0,i))

ax.set_xlim(-10, 340)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)
ax.view_init(elev=90)

def update(t):
    global quiver_dic
    for i in range(n):
        quiver_dic[i].remove()
        quiver_dic[i] = ax.quiver(*get_spin_vec(t,i))

ani = FuncAnimation(fig, update, frames= range(1000), interval=1)
#ani.save("multi_iniv.gif",writer='imagemagick')
plt.show()