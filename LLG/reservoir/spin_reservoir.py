import numpy as np
from scipy import linalg
from multi_spin.spin_wave import Spin_wave

class Spin_Reservoir:

    def __init__(self, inputs, num_input_nodes, num_reservoir_nodes, num_output_nodes,pulse, leak_rate=0.1, activator=np.tanh):
        self.inputs = inputs # 学習で使う入力
        self.log_reservoir_nodes = np.array([np.zeros(num_reservoir_nodes)]) # reservoir層のノードの状態を記録

        # init weights
        self.weights_input = self._generate_variational_weights(num_input_nodes, num_reservoir_nodes)
        self.weights_reservoir = self._generate_reservoir_weights(num_reservoir_nodes)
        self.weights_output = np.zeros([num_reservoir_nodes, num_output_nodes])

        # それぞれの層のノードの数
        self.num_input_nodes = num_input_nodes
        self.num_reservoir_nodes = num_reservoir_nodes
        self.num_output_nodes = num_output_nodes

        self.leak_rate = leak_rate # 漏れ率
        self.activator = activator # 活性化関数

        self.log_spin = []
        self.pulse = pulse

    # reservoir層のノードの次の状態を取得
    def _get_next_reservoir_nodes(self, input, current_state):
        next_state = (1 - self.leak_rate) * current_state
        next_state += self.leak_rate * (np.array([input]) @ self.weights_input
            + current_state @ self.weights_reservoir)
        return self.activator(next_state)

    # 出力層の重みを更新
    def _update_weights_output(self, lambda0):
        # Ridge Regression
        E_lambda0 = np.identity(self.num_reservoir_nodes) * lambda0 # lambda0
        inv_x = np.linalg.inv(self.log_reservoir_nodes.T @ self.log_reservoir_nodes + E_lambda0)
        #print(self.log_reservoir_nodes.T @ self.log_reservoir_nodes)
        # update weights of output layer
        #print(np.size(inv_x),np.size(self.log_reservoir_nodes),np.size(self.inputs))
        self.weights_output = (inv_x @ self.log_reservoir_nodes.T) @ self.inputs

    # 学習する
    def train(self, lambda0=0.5):
        self.get_spin()
        #print(self.log_spin[:,0].T[:3000])

        self.log_reservoir_nodes = self.log_spin[:,0].T[:3000]

        self._update_weights_output(lambda0)

    def get_spin(self):
        start_time = 0
        end_time = 50
        RATIO_TRAIN = 0.6
        t = [start_time, end_time]
        time_step = 5000
        t_eval = np.linspace(*t, time_step)
        LEAK_RATE = 0.02
        NUM_INPUT_SPIN = 1  # 入力スピンの数
        NUM_RESERVOIR_SPIN = 300  # スピンの数
        NUM_OUTPUT_NODES = 1  # アウトプットの数

        n = NUM_RESERVOIR_SPIN
        S0 = np.zeros((n, 3))

        for i in range(n):
            S0[i][2] = 10
        # print(S0)

        t = [start_time, end_time]  # t(時間)が0〜100まで動き、その時のfを求める。
        t_eval = np.linspace(*t, time_step)

        #print("here",self.pulse)


        spin = Spin_wave(0.0001, 0.01, [0.01, 0, 0], S0, t, t_eval, 0.1, 0.1, 1, self.pulse, n, 1)
        self.log_spin = spin.doit()



    # 学習で得られた重みを基に訓練データを学習できているかを出力
    def get_train_result(self):
        outputs = []
        reservoir_nodes = np.zeros(self.num_reservoir_nodes)
        for i,input in enumerate(self.inputs):
            outputs.append(self.get_output(self.log_reservoir_nodes[i]))
        return outputs

    # 予測する
    def predict(self, length_of_sequence, lambda0=0.01):
        predicted_outputs = [self.inputs[-1]] # 最初にひとつ目だけ学習データの最後のデータを使う
        reservoir_nodes = self.log_reservoir_nodes[-1] # 訓練の結果得た最後の内部状態を取得
        for i in range(length_of_sequence):
            reservoir_nodes = self.log_spin[:,0].T[i+3000]
            predicted_outputs.append(self.get_output(reservoir_nodes))
        return predicted_outputs[1:] # 最初に使用した学習データの最後のデータを外して返す

    # get output of current state
    def get_output(self, reservoir_nodes):
        # return self.activator(reservoir_nodes @ self.weights_output) 修正前
        return reservoir_nodes @ self.weights_output # 修正後

    #############################
    ##### private method ########
    #############################

    # 重みを0.1か-0.1で初期化したものを返す
    def _generate_variational_weights(self, num_pre_nodes, num_post_nodes):
        return (np.random.randint(0, 2, num_pre_nodes * num_post_nodes).reshape([num_pre_nodes, num_post_nodes]) * 2 - 1) * 0.1

    # Reservoir層の重みを初期化
    def _generate_reservoir_weights(self, num_nodes):
        weights = np.random.normal(0, 1, num_nodes * num_nodes).reshape([num_nodes, num_nodes])
        spectral_radius = max(abs(linalg.eigvals(weights)))
        return weights / spectral_radius