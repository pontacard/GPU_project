import numpy as np
from scipy import signal

class InputGenerator:

    def __init__(self, t_eval ,pulse):
        self.pulse = pulse
        self.t_eval = t_eval

    def generate_sin(self, amplitude=1.0):
        #t = np.linspace(self.start_time, self.end_time, self.num_time_steps, endpoint=False)
        #y = -(signal.square(2 * np.pi *0.1 * t)+1)/2


        new_pulse = np.empty
        pulse_dic = {}

        for t in self.t_eval:
            flag = 1
            for Bi in self.pulse:
                if Bi[0] <= t and Bi[1] >= t:
                    new_pulse = np.append(new_pulse, Bi[2])
                    pulse_dic[t] = Bi[2]
                    flag = 0
                    break
            if flag:
                new_pulse = np.append(new_pulse, [0, 0, 0])

        new_pulse = new_pulse[1:].reshape([-1, 3])

        #print(new_pulse[:,1])


        return new_pulse[:,1]
