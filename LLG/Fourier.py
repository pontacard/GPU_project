import numpy as np
import matplotlib.pyplot as plt

start = 0
end = 10
step = 100000
t  = np.linspace(start, end, step)
omega = 10
x = np.sin(omega * t)


N = len(x)  # サンプル数
f_s = 400  # サンプリングレート f_s[Hz] (任意)
dt = (end - start) / step # サンプリング周期 dt[s]

plt.plot(t,x)
plt.show()
y_fft = np.fft.fft(x)  # 離散フーリエ変換
freq = np.fft.fftfreq(N, d=dt)  # 周波数を割り当てる（※後述）
Amp = abs(y_fft / (N / 2))  # 音の大きさ（振幅の大きさ）
plt.plot(freq[1:int(N / 2)], Amp[1:int(N / 2)])  # A-f グラフのプロット
plt.show()