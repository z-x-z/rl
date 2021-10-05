'''
Description  : 演员-评论家算法(Actor-Critic)
Author       : CagedBird
Date         : 2021-08-10 21:17:19
FilePath     : /rl/src/rl_algorithms/A3C/simple_a3c.py
'''

import torch.multiprocessing as mp
from src.rl_algorithms.A3C.a2c import A2CNet
import torch
import gym

MAX_EPISODES = 1000
MAX_EPISODE_STEPS = 200
GLOBAL_UPDATE_ITERATION = 5


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # Adam优化器内部也有一些参数
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 1
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class GlobalNet(A2CNet):
    def __init__(self, n_state, n_action, lr, discount) -> None:
        super().__init__(n_state, n_action, discount)
        self.share_memory()  # 将网络的参数共享至内存中
        self.optim = SharedAdam(self.parameters(), lr=lr, betas=(0.95, 0.999))
        self.episode = mp.Value('i', 0)
        self.episode_reward = mp.Value('d', 0.)
        self.episode_reward_queue = mp.Queue()

    def record(self, episode_reward, worker_name):
        """ Record the episode and episode_reward from the worker."""
        with self.episode.get_lock():
            self.episode.value += 1
        with self.episode_reward.get_lock():
            if self.episode_reward.value == 0.:  # 初始值
                self.episode_reward.value = episode_reward
            else:
                self.episode_reward.value = self.episode_reward.value * 0.99 + episode_reward * 0.01
            self.episode_reward_queue.put(self.episode_reward.value)
        print("e{:3}, {} |er: {:5.0f} | g_er: {:5.0f}".format(self.episode.value, worker_name, episode_reward, self.episode_reward.value))


class Worker(mp.Process):
    def __init__(
        self,
        n_state,
        n_action,
        discount,
        global_net: GlobalNet,
        env: gym.Env,
        name="",
    ) -> None:
        super(Worker, self).__init__()
        self.global_net = global_net
        self.local_net = A2CNet(n_state, n_action, discount)
        self.env = env  # 每个worker与独立的env进行交互
        self.name = name
        self.buffer = None

    def zero_grad(self):
        for lp in self.local_net.parameters():
            if lp.grad is not None:
                lp.grad.zero_()

    def run(self):
        step_count = 1
        while self.global_net.episode.value < MAX_EPISODES:
            s = self.env.reset()
            episode_reward = 0
            for episode_step in range(MAX_EPISODE_STEPS):
                a = self.local_net.choose_action(torch.FloatTensor(s))
                a = a.clip(-2, 2)  # a的范围为[-2, 2]
                s_, r, done, _ = self.env.step(a)
                done = True if episode_step == MAX_EPISODE_STEPS - 1 else False
                episode_reward += r
                self.push(s, a, [(r + 8.1) / 8.1])  #  shrink the reward
                if step_count % GLOBAL_UPDATE_ITERATION == 0 or done:
                    self.update_global(s_, done)
                    if done:
                        self.global_net.record(episode_reward, self.name)
                s = s_
                step_count += 1

    def push(self, *parameters):
        """ parameters: (s, a, [r])  """
        if self.buffer is None:
            self.buffer = []
            [self.buffer.append([parameter]) for i, parameter in enumerate(parameters)]
        else:
            [self.buffer[i].append(parameter) for i, parameter in enumerate(parameters)]

    def update_global(self, s_, done):
        # Pytorch的网络输入的数据x若为2维的，则第一维表示数据的个数，第二维才是输入层的维度
        loss = self.local_net.loss(*[torch.FloatTensor(_) for _ in self.buffer], s_, done)
        self.local_net.zero_grad()  # 清空本地的梯度
        loss.backward()  # 得到本地网络的梯度
        for gp, lp in zip(self.global_net.parameters(), self.local_net.parameters()):
            gp._grad = lp.grad
        self.global_net.optim.step()  # 利用本地网络的梯度对全局网络进行更新
        self.buffer = None  # 清理buff
        self.local_net.load_state_dict(self.global_net.state_dict())  # 本地网络重新获取全局网络的参数


class SimpleA3C:
    def __init__(self, lr, discount, worker_envs) -> None:
        n_state = worker_envs[0].observation_space.shape[0]  # 4个状态量
        n_action = worker_envs[0].action_space.shape[0]  # 2种动作
        self.global_net = GlobalNet(n_state, n_action, lr, discount)
        self.work_nets = [
            Worker(n_state, n_action, discount, self.global_net, env, name="w{}".format(i)) for i, env in enumerate(worker_envs)
        ]

    def run(self):
        [work.start() for work in self.work_nets]
        [work.join() for work in self.work_nets]

