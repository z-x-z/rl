from os import SEEK_CUR
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# DQN网络输出的是所有动作的Q值表，而Q值表的大小有限，所以可选择的动作也是有限的（离散）
class DQNNet(nn.Module):
    def __init__(self, N_in, N_out, N_HIDDEN_LAYERS=128):
        super(DQNNet, self).__init__()
        self.fc1 = nn.Linear(N_in, N_HIDDEN_LAYERS)
        self.relu1 = nn.ReLU()
        self.out = nn.Linear(N_HIDDEN_LAYERS, N_out)

        self.fc1.weight.data.normal_(0, 0.1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        action_values = self.out(x)
        return action_values


class SimpleDeepQNetwork:
    def __init__(self,
                 n_features,
                 n_actions,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 e_greedy_increment=None,
                 replace_target_iterations=300,
                 memory_size=500,
                 batch_size=32,
                 use_gpu=False):
        self.memory_count = 0
        self.n_features = n_features
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.e_greedy_max = e_greedy
        self.e_greedy_increment = e_greedy_increment
        self.e_greedy = 0 if e_greedy_increment is not None else e_greedy
        self.replace_target_iterations = replace_target_iterations
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        self.memory = np.zeros((self.memory_size, 2 * n_features + 2))  # (memory_size, 2 * n_features + 2)
        self.learn_step_count = 0
        self.loss_list = []
        self.__build_net()

    def __build_net(self):
        # 网络输出的是q值，而非直接选择动作
        self.eval_net = DQNNet(self.n_features, self.n_actions)
        self.target_net = DQNNet(self.n_features, self.n_actions)
        self.loss_func = nn.MSELoss()
        if self.use_gpu:
            self.eval_net = self.eval_net.cuda()  # 将网络模型转移至gpu中
            self.target_net = self.target_net.cuda()
            self.loss_func = self.loss_func.cuda()  # 将损失函数转移到gpu中
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)  # 优化器不能转移到gpu中

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_count % self.memory_size
        self.memory[index, :] = transition
        self.memory_count += 1

    def choose_action(self, state, is_train=True):
        # 如果是训练模式，使用epsilon greedy算法
        if is_train and np.random.uniform() < self.e_greedy:
            action = np.random.randint(0, self.n_actions)
        # 若不是训练模式则使用贪婪模式
        else:
            x = torch.unsqueeze(torch.FloatTensor(state), 0)
            if self.use_gpu:
                x = x.cuda()
            action_values = self.eval_net(x)
            action = torch.argmax(action_values).item()  # 选取所有动作值中最大的动作
        return action

    def learn(self):
        self.learn_step_count += 1
        if self.learn_step_count % self.replace_target_iterations == 0:  # 每过一定步长才更新target_net的参数
            self.target_net.load_state_dict(self.eval_net.state_dict())

        "抽样"
        sample_indices = np.random.choice(self.memory_size, self.batch_size)  # (32)
        batch_memory = self.memory[sample_indices, :]  # 32 * 10
        b_s = torch.FloatTensor(batch_memory[:, :self.n_features])  # 32 * 4
        b_a = torch.LongTensor(batch_memory[:, self.n_features:self.n_features + 1].astype(int))  # 32 * 1
        b_r = torch.FloatTensor(batch_memory[:, self.n_features + 1:self.n_features + 2])  # 32 * 1
        b_s_ = torch.FloatTensor(batch_memory[:, -self.n_features:])  # 32 * 4
        if self.use_gpu:
            b_s, b_a, b_r, b_s_ = [_.cuda() for _ in [b_s, b_a, b_r, b_s_]]
        # gpu中的模型只能操作gpu中的数据
        "计算误差"
        q_eval = self.eval_net(b_s).gather(1, b_a)  # 32 * 1 预测值
        q_next = self.target_net(b_s_).detach()  # 32 * 2 detach() 取消其反向传播
        q_target = b_r + q_next.max(dim=1)[0].view(self.batch_size, -1) * self.reward_decay  # 32 * 1 真实值
        loss = self.loss_func(q_eval, q_target)
        self.loss_list.append(loss.item())

        "误差反向传播"
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        "更新epsilon"
        if self.e_greedy_increment:
            self.e_greedy += self.e_greedy_increment
            if self.e_greedy > self.e_greedy_max:
                self.e_greedy = self.e_greedy_max

    def plot_loss(self):
        plt.plot(list(range(len(self.loss_list))), self.loss_list)
        plt.show()
