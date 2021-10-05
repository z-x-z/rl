'''
Description  : 演员-评论家算法(Actor-Critic)
Author       : CagedBird
Date         : 2021-08-10 21:17:19
FilePath     : /rl/src/rl_algorithms/AC/simple_ac.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)


# Actor - policy net
class Actor(nn.Module):
    def __init__(self, N_in, N_out, learning_rate=0.01, N_HIDDEN_LAYERS=32):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(N_in, N_HIDDEN_LAYERS)
        self.relu1 = nn.ReLU()
        self.out = nn.Linear(N_HIDDEN_LAYERS, N_out)
        initialize_weights(self)
        # other attributes
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        action_probs = F.softmax(self.out(x), dim=0)
        return action_probs

    def learn(self, observation, action, td_error):
        observation = torch.FloatTensor(observation)
        action = torch.LongTensor([action])  # 注意要在action上加方括号
        log_action_prob = torch.log(self(observation)[action])
        loss = log_action_prob * td_error  # 不需要加负号
        # 梯度下降法使得演员的策略网络朝着td误差减小的方向收敛
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Critic - value net
class Critic(nn.Module):
    # Critic网络是输出的价值函数，所以输出的是一维
    def __init__(self, N_in, learning_rate=0.01, discount=0.9, N_HIDDEN_LAYERS=32):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(N_in, N_HIDDEN_LAYERS)
        self.relu1 = nn.ReLU()
        self.out = nn.Linear(N_HIDDEN_LAYERS, 1)
        initialize_weights(self)
        # other attributes
        self.discount = discount
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        action_values = self.out(x)
        return action_values

    def learn(self, observation, next_observation, reward):
        observation = torch.FloatTensor(observation)
        next_observation = torch.FloatTensor(next_observation)
        v, v_ = self(observation), self(next_observation)
        critic_loss = self.loss_func(reward + self.discount * v_, v)
        # 梯度下降法使得评价家的值函数网络向着td误差减小的方向收敛
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            td_error = reward + self.discount * v_ - v
        return td_error


class SimpleAC:
    def __init__(self, n_state, n_action, lr_a=1e-3, lr_c=1e-2, discount=0.9) -> None:
        self.n_features = n_state
        self.n_actions = n_action
        self.discount = discount
        self.__build_net(lr_a, lr_c)

    def __build_net(self, lr_a, lr_c):
        self.actor = Actor(self.n_features, self.n_actions, lr_a)
        self.critic = Critic(self.n_features, lr_c)

    def choose_action(self, observation):
        # 使用actor进行动作选择
        with torch.no_grad():
            observation = torch.FloatTensor(observation)
            action_probs = self.actor(observation).numpy()
            action = np.random.choice(action_probs.shape[0], p=action_probs)
        return action

    def learn(self, observation, next_observation, action, reward):
        # G = (Q(s, a) - V(s)), Q(s, a) = E(r + discount * V(s+1)) 
        # => G = (r + discount * V(s+1) - V(s)) <=> G = td_error
        # 先在critic中计算td误差，并利用td误差对网络进行更新，并返回td误差值
        td_error = self.critic.learn(observation, next_observation, reward)
        # 利用td误差值对
        self.actor.learn(observation, action, td_error)
