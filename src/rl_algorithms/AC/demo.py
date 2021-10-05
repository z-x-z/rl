"""
@ Author: Peter Xiao
@ Date: 2020/7/23
@ Filename: Actor_critic.py
@ Brief: 使用 Actor-Critic算法训练CartPole-v0
————————————————
版权声明：本文为CSDN博主「_Epsilon_」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_34003876/article/details/107477426
"""
import gym
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# Hyper Parameters for Actor
GAMMA = 0.95  # discount factor
LR = 0.01  # learning rate

# Use GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
torch.backends.cudnn.enabled = False  # 非确定性算法


class PGNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PGNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, action_dim)
        self.initialize_weights()

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                nn.init.constant_(m.bias.data, 0.01)


class Actor(object):
    # dqn Agent
    def __init__(self, env):  # 初始化
        # 状态空间和动作空间的维度
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # init network parameters
        self.network = PGNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)

        # init some parameters
        self.time_step = 0

    def choose_action(self, observation):
        observation = torch.FloatTensor(observation).to(device)
        network_output = self.network.forward(observation)
        with torch.no_grad():
            # prob_weights = F.softmax(network_output, dim=0).cuda().data.cpu().numpy()
            prob_weights = F.softmax(network_output, dim=0).detach().numpy()
        action = np.random.choice(range(prob_weights.shape[0]),
                                  p=prob_weights)  # select action w.r.t the actions prob
        return action

    def learn(self, state, action, td_error):
        self.time_step += 1
        # Step 1: 前向传播
        softmax_input = self.network.forward(torch.FloatTensor(state).to(device)).unsqueeze(0)
        action = torch.LongTensor([action]).to(device)
        neg_log_prob = F.cross_entropy(input=softmax_input, target=action, reduction='none')

        # Step 2: 反向传播
        # 这里需要最大化当前策略的价值，因此需要最大化neg_log_prob * tf_error,即最小化-neg_log_prob * td_error
        loss_a = neg_log_prob * td_error
        self.optimizer.zero_grad()
        loss_a.backward()
        self.optimizer.step()


# Hyper Parameters for Critic
EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch
REPLACE_TARGET_FREQ = 10  # frequency to update target Q network


class QNetwork(nn.Module):
    def __init__(self, state_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, 1)   # 这个地方和之前略有区别，输出不是动作维度，而是一维
        self.initialize_weights()

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                nn.init.constant_(m.bias.data, 0.01)


class Critic(object):
    def __init__(self, env):
        # 状态空间和动作空间的维度
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # init network parameters
        self.network = QNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

        # init some parameters
        self.time_step = 0
        self.epsilon = EPSILON  # epsilon值是随机不断变小的

    def train_Q_network(self, state, reward, next_state):
        s, s_ = torch.FloatTensor(state).to(device), torch.FloatTensor(next_state).to(device)
        # 前向传播
        v = self.network.forward(s)     # v(s)
        v_ = self.network.forward(s_)   # v(s')

        # 反向传播
        loss_q = self.loss_func(reward + GAMMA * v_, v)
        self.optimizer.zero_grad()
        loss_q.backward()
        self.optimizer.step()

        with torch.no_grad():
            td_error = reward + GAMMA * v_ - v

        return td_error


# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 3000  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode


def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    actor = Actor(env)
    critic = Critic(env)
    train_total_reward = 0

    for episode in range(EPISODE):
        # initialize task
        state = env.reset()
        # Train
        for step in range(STEP):
            action = actor.choose_action(state)  # SoftMax概率选择action
            next_state, reward, done, _ = env.step(action)
            train_total_reward += reward
            td_error = critic.train_Q_network(state, reward, next_state)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(state, action, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            state = next_state
            if done:
                break

        # Test every 100 episodes
        if episode % 100 == 0:
            train_average_reward = train_total_reward / 100
            train_total_reward = 0
            test_total_reward = 0
            for _ in range(TEST):
                state = env.reset()
                for _ in range(STEP):
                    action = actor.choose_action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    test_total_reward += reward
                    if done:
                        break
            test_ave_reward = test_total_reward/TEST
            print('Episode: {}. Train average reward: {}. Test average reward: {}.'.format(episode,  train_average_reward, test_ave_reward))


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('Total time is ', time_end - time_start, 's')

