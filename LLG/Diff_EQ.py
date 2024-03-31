import scipy as sc
import numpy as np
import matplotlib.pyplot as plt


def f(t, x, a):  # 関数を設定
    dxdt = a * x  # 要するにxの時間微分がa*x(解はx=Ce^at)
    return dxdt


x0 = 1
t = np.arange(0, 3, 0.01)  # t(時間)が0〜3まで動き、その時のfを求める。

y = sc.integrate.odeint(f, x0, t, args=(1,))

print(y)
plt.plot(t, y)
plt.xlabel('t')
plt.ylabel('y')

plt.show()
