'''
Description  : 
Author       : CagedBird
Date         : 2021-08-20 10:36:56
FilePath     : /rl/src/rl_algorithms/DDPG/simple_ddpg.py
'''
import torch
import torch.nn as nn
import numpy as np


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            # m.weight.data.normal_(0, 0.1)
            nn.init.normal_(m.weight.data, 0, 0.1)
            # nn.init.constant_(m.bias.data, 0.01)


# 策略网络
# 输入：状态值(state_dim)
# 输出：动作(action_dim)
class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, N_HIDDEN1=32) -> None:
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, N_HIDDEN1)
        self.out = nn.Linear(N_HIDDEN1, action_dim)
        initialize_weights(self)

    def forward(self, state):
        action = torch.relu(self.fc1(state))
        action = torch.tanh(self.out(action))
        # action = 2 * action
        # 若给定了动作各维度值的范围，则先进行用Tanh映射到[-1, 1]空间，其后再映射到[a, b]空间中
        return action


class Actor():
    def __init__(self, state_dim, action_dim, action_bound, lr, tau) -> None:
        self.online_net = ActorNet(state_dim, action_dim)
        self.target_net = ActorNet(state_dim, action_dim)
        self.action_dim = action_dim
        self.action_bound = action_bound  # Ndarray(2, action_dim) [lows ,highs]
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)  # 使用
        self.tau = tau

    def soft_update(self):
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1 - self.tau) * target_param)

    def choose_action(self, state):
        action = self.online_net(torch.FloatTensor(state)).detach().numpy()
        # 将0-1空间的动作，映射到a-b空间中
        for i in range(self.action_bound.shape[1]):
            action[i] = action[i] * (self.action_bound[1, i] - self.action_bound[0, i]) / 2 + (self.action_bound[1, i] +
                                                                                               self.action_bound[0, i]) / 2
        return action  # 返回的形如(action_dim)

    def learn(self, b_s, critic):
        predict_a = self.online_net(b_s)
        Q = critic.online_net(b_s, predict_a)  # Q(s, C(s))
        actor_loss = -torch.mean(Q)  # 负号表示用梯度上升法使得Q值增大
        self.optimizer.zero_grad()
        # actor的梯度等于Q值的梯度
        actor_loss.backward()
        self.optimizer.step()


# Q价值网络
# 输入：状态，动作 (state_dim, action_dim)
# 输出Q(s, a)值 (1)
class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, N_HIDDEN1=32) -> None:
        super(CriticNet, self).__init__()
        """ self.fc1 = nn.Linear(state_dim + action_dim, N_HIDDEN1)
        self.out = nn.Linear(N_HIDDEN1, 1) """
        self.fcs = nn.Linear(state_dim, N_HIDDEN1)
        self.fca = nn.Linear(action_dim, N_HIDDEN1)
        self.out = nn.Linear(N_HIDDEN1, 1)
        initialize_weights(self)

    def forward(self, s, a):
        """ input = torch.cat((s, a), dim=1)
        Q = self.out(torch.relu(self.fc1(input))) """
        Q = self.out(torch.relu(self.fcs(s) + self.fca(a)))
        return Q


class Critic():
    def __init__(self, state_dim, action_dim, lr, tau, discount) -> None:
        self.online_net = CriticNet(state_dim, action_dim)
        self.target_net = CriticNet(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)  # 使用
        self.tau = tau
        self.discount = discount
        self.loss_func = nn.MSELoss()

    def soft_update(self):
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1 - self.tau) * target_param)

    def learn(self, b_s, b_a, b_r, b_s_, actor):
        b_a_ = actor.target_net(b_s_)  # 通过actor在线网络预测下一个状态将采取的动作
        Q_ = self.target_net(b_s_, b_a_)
        target_Q = b_r + self.discount * Q_
        Q = self.online_net(b_s, b_a)
        critic_loss = self.loss_func(target_Q, Q)
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()


class Memory():
    def __init__(self, capacity, state_dim, action_dim) -> None:
        self.capacity = capacity
        self.index = 0
        self.size = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        # [s, a, r, s_]
        self.memory = np.empty((self.capacity, 2 * state_dim + action_dim + 1))  # (n, m)

    def store(self, s, a, r, s_):
        # np.hstack将多个array按第二个维度（除了array是一维）堆叠起来
        transition = np.hstack((s, a, [r], s_))
        self.memory[self.index, :] = transition
        self.size += 1
        self.index = self.size % self.capacity

    def sample(self, batch_size):
        batch_indices = np.random.choice(range(self.capacity), batch_size)
        b_s = self.memory[batch_indices, :self.state_dim]
        b_a = self.memory[batch_indices, self.state_dim:self.state_dim + self.action_dim]
        b_r = self.memory[batch_indices, self.state_dim + self.action_dim:self.state_dim + self.action_dim + 1]
        b_s_ = self.memory[batch_indices, -self.state_dim:]
        return b_s, b_a, b_r, b_s_


class SimpleDDPG():
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_bound=None,
                 lr_a=1e-3,
                 lr_c=1e-3,
                 discount=0.95,
                 memory_capacity=1000,
                 batch_size=32,
                 tau=1e-2,
                 noise_init_var=3,
                 noise_decay=0.9995):
        #  考虑用setattr方法重新实现
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.discount = discount
        self.batch_size = batch_size
        self.tau = tau
        self.noise_var = noise_init_var
        self.noise_decay = noise_decay
        self.actor = Actor(state_dim, action_dim, action_bound, lr_a, tau)
        self.critic = Critic(state_dim, action_dim, lr_c, tau, discount)
        self.memory = Memory(memory_capacity, state_dim, action_dim)

    def add_noise(self, action):
        #  (state_dim) -np.expend_dims(0)-> (1, state_dim) -squeeze(0)-> (state_dim)
        action = np.random.normal(np.expand_dims(action, 0), self.noise_var)
        return np.clip(action, self.action_bound[0], self.action_bound[1]).squeeze(0)

    def choose_action(self, state) -> int:
        return self.actor.choose_action(state)

    def learn(self, s, a, r, s_) -> None:
        self.memory.store(s, a, r, s_)
        if self.memory.size >= self.memory.capacity:
            self.noise_var *= self.noise_decay
            # 将batch_memory拆分，并且转化为torch.Tensor形式
            b_s, b_a, b_r, b_s_ = [torch.FloatTensor(_) for _ in self.memory.sample(self.batch_size)]
            self.actor.learn(b_s, self.critic)
            self.critic.learn(b_s, b_a, b_r, b_s_, self.actor)
            self.actor.soft_update()
            self.critic.soft_update()