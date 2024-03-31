import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation

n = 50
count = 0

def S(S,t):  # 関数を設定
    global count
    dSdt = np.empty(0)
    ba, fo = 0,0



    for i in range(n):
        Si = [S[3 * i + 0], S[3 * i + 1], S[3 * i + 2]]
        Sib = []
        Sif = []
        if i == 0:
            fo = i + 1
            Sib = [0,0,0]
            Sif = [S[3 * fo + 0], S[3 * fo + 1], S[3 * fo + 2]]
            #sigma_pulse = np.sin(100 * (t - 10)) / (3.14 * (t - 10)) + np.sin(100 * (t - 20)) / (3.14 * (t - 20)) + np.sin(100 * (t - 30)) / (3.14 * (t - 30))+np.sin(100 * (t - 40)) / (3.14 * (t - 40))+np.sin(100 * (t - 50)) / (3.14 * (t - 50))+np.sin(100 * (t - 60)) / (3.14 * (t - 60))+np.sin(100 * (t - 70)) / (3.14 * (t - 70))+np.sin(100 * (t - 80)) / (3.14 * (t - 80))
            #sigma_pulse = np.sin(200 * (t - 1200)) / (3.14 * (t - 1200))
            B = [0, 0, 0.02]
            ScrSSB = [(Si[1] * (Sib[2] + Sif[2] + B[2]) - Si[2] * (Sib[1] + Sif[1] + +B[1])),
                      (Si[2] * (Sib[0] + Sif[0] + B[0]) - Si[0] * (Sib[2] + Sif[2] + B[2])),
                      (Si[0] * (Sib[1] + Sif[1] + B[1]) - Si[1] * (Sib[0] + Sif[0] + B[0]))]

            Snorm = np.linalg.norm(Si)

            dSixdt = 2 * ScrSSB[0] - ((0000.1 / Snorm) * (Si[1] * ScrSSB[2] - Si[2] * ScrSSB[1]))
            dSiydt = 2 * ScrSSB[1] - ((0000.1 / Snorm) * (Si[2] * ScrSSB[0] - Si[0] * ScrSSB[2]))
            dSizdt = 2 * ScrSSB[2] - ((0000.1 / Snorm) * (Si[0] * ScrSSB[1] - Si[1] * ScrSSB[0]))

            dSdt = np.append(dSdt, [dSixdt, dSiydt, dSizdt])

        elif i == n-1:
            ba = i - 1
            Sib = [S[3 * ba + 0], S[3 * ba + 1], S[3 * ba + 2]]
            Sif = [0,0,0]
            B = [0, 0, 0.02]
            ScrSSB = [(Si[1] * (Sib[2] + Sif[2] + B[2]) - Si[2] * (Sib[1] + Sif[1] + +B[1])),
                      (Si[2] * (Sib[0] + Sif[0] + B[0]) - Si[0] * (Sib[2] + Sif[2] + B[2])),
                      (Si[0] * (Sib[1] + Sif[1] + B[1]) - Si[1] * (Sib[0] + Sif[0] + B[0]))]

            Snorm = np.linalg.norm(Si)

            dSixdt = 2 * ScrSSB[0] - ((0000.1 / Snorm) * (Si[1] * ScrSSB[2] - Si[2] * ScrSSB[1]))
            dSiydt = 2 * ScrSSB[1] - ((0000.1 / Snorm) * (Si[2] * ScrSSB[0] - Si[0] * ScrSSB[2]))
            dSizdt = 2 * ScrSSB[2] - ((0000.1 / Snorm) * (Si[0] * ScrSSB[1] - Si[1] * ScrSSB[0]))

            dSdt = np.append(dSdt, [dSixdt, dSiydt, dSizdt])


        else:
            fo = i + 1
            ba = i - 1
            Sib = [S[3 * ba + 0], S[3 * ba + 1], S[3 * ba + 2]]
            Sif = [S[3 * fo + 0], S[3 * fo + 1], S[3 * fo + 2]]


            B = [0, 0, 0.02]
            ScrSSB = [(Si[1] * (Sib[2] + Sif[2] + B[2]) - Si[2] * (Sib[1] + Sif[1] + +B[1])),
                      (Si[2] * (Sib[0] + Sif[0] + B[0]) - Si[0] * (Sib[2] + Sif[2] + B[2])),
                      (Si[0] * (Sib[1] + Sif[1] + B[1]) - Si[1] * (Sib[0] + Sif[0] + B[0]))]

            Snorm = np.linalg.norm(Si)

            dSixdt = 2 * ScrSSB[0] - ((0000.1 / Snorm) * (Si[1] * ScrSSB[2] - Si[2] * ScrSSB[1]))
            dSiydt = 2 * ScrSSB[1] - ((0000.1 / Snorm) * (Si[2] * ScrSSB[0] - Si[0] * ScrSSB[2]))
            dSizdt = 2 * ScrSSB[2] - ((0000.1 / Snorm) * (Si[0] * ScrSSB[1] - Si[1] * ScrSSB[0]))

            dSdt = np.append(dSdt, [dSixdt, dSiydt, dSizdt])
    dSdt = np.reshape(dSdt,(1,-1))

    return dSdt[0]

S0 = np.zeros((n,3))

S0[0][1] = 1
for i in range(n):
    if i == 0:
        continue
    S0[i][2] = 1


#S0 = np.array([[1,0,0],[0,0,1],[0,0,1]])

S0 = np.reshape(S0,(1,-1))
print(S0)


t = np.linspace(0, 1000, 1000)   # t(時間)が0〜3まで動き、その時のfを求める。

y = sc.integrate.odeint(S, S0[0], t)
print(y)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

frames = []

S = np.reshape(y,(-1,n,3))

print(S)

def get_spin_vec(t,i):
    Ox,Oy,Oz = 3*i,0,0
    x = S[t][i][0]
    y = S[t][i][1]
    z = S[t][i][2]
    return Ox, Oy, Oz, x, y, z

quiver_dic = dict()
for i in range(n):
    quiver_dic[i] = ax.quiver(*get_spin_vec(0,i))

ax.set_xlim(-10, 333)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)
ax.view_init(elev=90)

def update(t):
    global quiver_dic
    for i in range(n):
        quiver_dic[i].remove()
        quiver_dic[i] = ax.quiver(*get_spin_vec(t,i))

ani = FuncAnimation(fig, update, frames= range(1000), interval=1)
#ani.save("200_spin3.gif",writer='imagemagick')
plt.show()