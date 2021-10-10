# coding=UTF-8
'''
Description  : 
Author       : caged_bird
Date         : 2021/1/22 3:10 PM
File         : rl - simple_policy_gradients
'''
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class PolicyGradientsNet(nn.Module):
    def __init__(self, N_in, N_out, N_HIDDEN_LAYERS=128):
        super(PolicyGradientsNet, self).__init__()
        self.fc1 = nn.Linear(N_in, N_HIDDEN_LAYERS)
        self.relu1 = nn.ReLU()
        self.out = nn.Linear(N_HIDDEN_LAYERS, N_out)
        self.softmax = nn.Softmax(1)  # 1 - 表示沿着轴1计算

        self.fc1.weight.data.normal_(0, 0.1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        # softmax用于输出各个动作的概率
        action_probs = self.softmax(self.out(x))
        return action_probs


class SimplePolicyGradients:
    def __init__(self,
                 n_features,
                 n_actions,
                 learning_rate=0.01,
                 discount=0.9):
        self.n_features = n_features
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount = discount

        self.__build_net()
        self.trajectory_states = []
        self.trajectory_actions = []
        self.trajectory_rewards = []
        self.trajectory_log_probs = []
        self.loss_record = []

    def __build_net(self):
        # 网络输出的是各动作的概率值
        self.policy_net = PolicyGradientsNet(self.n_features, self.n_actions)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def store_transition(self, state, action, reward):
        self.trajectory_states.append(state)
        self.trajectory_actions.append(action)
        self.trajectory_rewards.append(reward)

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)  # (1, trajectory_size)
        action_probs = self.policy_net(state)  # (N_A, trajectory_size)
        m = Categorical(action_probs)  # 生成分布
        action = m.sample()  # 进行采样
        self.trajectory_log_probs.append(m.log_prob(action))
        return action.item()  # 返回的值而非tensor([1])

    def learn(self):
        eps = 1e-5
        gains = np.empty(len(self.trajectory_rewards), dtype=np.float)
        gains[-1] = self.trajectory_rewards[-1]
        for i in range(len(self.trajectory_rewards) - 2, -1, -1):
            gains[i] = self.trajectory_rewards[i] + self.discount * gains[i + 1]
        gains = torch.from_numpy(gains)
        gains = (gains - gains.mean()) / (gains.std() + eps)  # 归一化,使得gains有正有负
        policy_loss = []
        for log_prob, gain in zip(self.trajectory_log_probs, gains):
            policy_loss.append(-log_prob * gain)

        "反向传播"
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()  # 误差求和
        self.loss_record.append(policy_loss.item())
        policy_loss.backward()  # 误差后向传递
        self.optimizer.step()  # 梯度下降

        "清理数据"
        del self.trajectory_log_probs[:]
        del self.trajectory_rewards[:]


