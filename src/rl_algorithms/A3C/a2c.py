'''
Description  : 
Author       : CagedBird
Date         : 2021-09-14 21:23:47
FilePath     : /rl/src/rl_algorithms/A3C/a2c.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.)


# Actor - policy net use normal distribution
class ActorNet(nn.Module):
    def __init__(self, N_in, N_out, N_HIDDEN_LAYERS=200):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(N_in, N_HIDDEN_LAYERS)
        self.normal_mean = nn.Linear(N_HIDDEN_LAYERS, N_out)
        self.normal_std = nn.Linear(N_HIDDEN_LAYERS, N_out)
        initialize_weights(self)
        self.distribution = torch.distributions.normal.Normal  # 选则正太分布作为近似概率分布函数
        self.entropy_coef = 0.005

    def forward(self, x):
        x = F.relu6(self.fc1(x))
        mean = 2 * F.tanh((self.normal_mean(x)))  # 由于动作的范围为[-2, 2]，所以均值范围在[-2, 2]
        std = F.softplus(self.normal_std(x)) + 1e-3  # std应该大于始终0
        return mean, std

    def choose_action(self, s):
        mean, std = self(s)
        m = self.distribution(mean, std)
        return m.sample()

    def loss(self, b_s, b_a, td):
        mean, std = self(b_s)
        m = self.distribution(mean, std)
        log_action_prob = m.log_prob(b_a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)
        except_v = log_action_prob * td + self.entropy_coef * entropy
        loss = -except_v  # 不需要加负号
        return loss


# Critic - value net
class CriticNet(nn.Module):
    # Critic网络是输出的价值函数，所以输出的是一维
    def __init__(self, N_in, discount=0.9, N_HIDDEN_LAYERS=100):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(N_in, N_HIDDEN_LAYERS)
        self.relu1 = nn.ReLU()
        self.out = nn.Linear(N_HIDDEN_LAYERS, 1)
        initialize_weights(self)
        self.discount = discount
        # 这里A3C推荐使用RMSProp
        self.loss_func = nn.MSELoss()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        action_values = self.out(x)
        return action_values

    def loss_and_td(self, b_s, b_v_target):
        b_v = self(b_s)
        td = b_v_target - b_v
        loss = td.pow(2)
        return loss, td.detach()


# 需要让A2C也继承nn.Module，以使得通过A2C对象能够获取该对象下actor和critic网络的参数信息
class A2CNet(nn.Module):
    def __init__(self, n_state, n_action, discount=0.9) -> None:
        super(A2CNet, self).__init__()
        self.n_features = n_state
        self.n_actions = n_action
        self.discount = discount
        self.__build_net()

    def __build_net(self):
        self.actor = ActorNet(self.n_features, self.n_actions)
        self.critic = CriticNet(self.n_features)

    def choose_action(self, state):
        # 使用actor进行动作选择
        with torch.no_grad():
            action = self.actor.choose_action(state).numpy()
        return action

    def loss(self, b_s, b_a, b_r, s_, done):
        v_s_ = 0 if done else self.critic(torch.FloatTensor([s_])).item()  # [1, 1]
        b_v_target = []
        for r in np.nditer(b_r.numpy()):  # b_r: [n, 1]
            v_s_ = r + self.discount * v_s_
            b_v_target.append([v_s_])
        b_v_target.reverse()
        b_v_target = torch.FloatTensor(b_v_target)  # [n, 1]
        # 先在critic中计算td误差，并利用td误差对网络进行更新，并返回td误差值
        critic_loss, td = self.critic.loss_and_td(b_s, b_v_target)
        # 利用td误差值对
        actor_loss = self.actor.loss(b_s, b_a, td)
        return (critic_loss + actor_loss).mean()
