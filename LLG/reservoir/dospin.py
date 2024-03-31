import numpy as np
from input_gen import InputGenerator
from spin_reservoir import Spin_Reservoir
import matplotlib.pyplot as plt

start_time = 0
end_time = 50
RATIO_TRAIN = 0.6
t = [start_time, end_time]
time_step = 5000
t_eval = np.linspace(*t, time_step)
LEAK_RATE=0.02
NUM_INPUT_SPIN = 1  #入力スピンの数
NUM_RESERVOIR_SPIN = 300  #スピンの数
NUM_OUTPUT_NODES = 1    #アウトプットの数


pulse = []
Num_pulse = 10
#print(t_eval)

for i in range(time_step):
    #tempul = [0.03*i, 0.03*i + 0.03, [0, 100 * np.sin(0.08*i), 0]]  #パルス波の作成
    tempul = [1 + 3 * i, 3 * i + 2, [0, -100, 0]]  # パルス波の作成
    pulse.append(tempul)




# example of activator
def ReLU(x):
    return np.maximum(0, x)

def main():
    i_gen = InputGenerator(t_eval, pulse)
    data = i_gen.generate_sin(amplitude=1)  #pulseを元にtimestep分の要素のある入力配列を作る
    num_train = int(len(data) * RATIO_TRAIN)
    train_data = data[:num_train]       #学習データを取り分けている
    #print(train_data)

    model = Spin_Reservoir(inputs=train_data,
        num_input_nodes=NUM_INPUT_SPIN,
        num_reservoir_nodes=NUM_RESERVOIR_SPIN,
        num_output_nodes=NUM_OUTPUT_NODES,
        leak_rate=LEAK_RATE,
        pulse = pulse)

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