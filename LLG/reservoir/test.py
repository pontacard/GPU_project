import numpy as np

pulse = []
Num_pulse = 30
t = [0,30]  # t(時間)が0〜100まで動き、その時のfを求める。
t_eval = np.linspace(*t, 300)
for i in range(Num_pulse):
    tempul = [i,i+0.2,[0,-1,0]]
    pulse.append(tempul)

print(pulse[0][0])

print(t_eval)
new_pulse = np.empty
pulse_dic = {}

for t in t_eval:
    flag = 1
    for Bi in pulse:
        if Bi[0] <= t and Bi[1] >= t:
            new_pulse = np.append(new_pulse,Bi[2])
            pulse_dic[t] = Bi[2]
            flag = 0
            break
    if flag:
        new_pulse = np.append(new_pulse,[0,0,0])


print(new_pulse[1:].reshape([-1,3]))
print(new_pulse[0])
print(pulse_dic.values())

