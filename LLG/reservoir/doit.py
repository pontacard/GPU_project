import numpy as np
from input_gen import InputGenerator
from reservoir_computing import ReservoirNetWork
import matplotlib.pyplot as plt

start_time = 0
end_time = 100
RATIO_TRAIN = 0.6
t = [start_time, end_time]
time_step = 1000
t_eval = np.linspace(*t, time_step)
LEAK_RATE=0.02
NUM_INPUT_SPIN = 1  #入力スピンの数
NUM_RESERVOIR_SPIN = 100  #スピンの数
NUM_OUTPUT_NODES = 1    #アウトプットの数


pulse = []
Num_pulse = 30
print(t_eval)

for i in range(Num_pulse):
    tempul = [10*i, 10*i + 5, [0, -300, 0]]  #パルス波の作成
    pulse.append(tempul)




# example of activator
def ReLU(x):
    return np.maximum(0, x)

def main():
    i_gen = InputGenerator(t_eval, pulse)
    data = i_gen.generate_sin(amplitude=1)
    num_train = int(len(data) * RATIO_TRAIN)
    train_data = data[:num_train]
    #print(train_data)

    model = ReservoirNetWork(inputs=train_data,
        num_input_nodes=NUM_INPUT_SPIN,
        num_reservoir_nodes=NUM_RESERVOIR_SPIN,
        num_output_nodes=NUM_OUTPUT_NODES,
        leak_rate=LEAK_RATE)

    model.train() # 訓練
    train_result = model.get_train_result() # 訓練の結果を取得

    num_predict = int(len(data[num_train:]))
    predict_result = model.predict(num_predict)

    t = t_eval
    ## plot
    plt.plot(t, data, label="inputs")
    plt.plot(t[:num_train], train_result, label="trained")
    plt.plot(t[num_train:], predict_result, label="predicted")
    plt.axvline(x=int(end_time * RATIO_TRAIN), label="end of train", color="green") # border of train and prediction
    plt.legend()
    plt.title("Echo State Network Sin Prediction")
    plt.xlabel("time[ms]")
    plt.ylabel("y(t)")
    plt.show()

if __name__=="__main__":
    main()
