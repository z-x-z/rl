'''
Description  : 
Author       : CagedBird
Date         : 2021-08-13 15:19:51
FilePath     : /rl/src/rl_algorithms/A3C/test_a3c.py
'''

import gym
from src.rl_algorithms.A3C.simple_a3c import SimpleA3C
import torch.multiprocessing as mp
import matplotlib.pyplot as plt


def test_a3c(a3c: SimpleA3C):
    a3c.run()
    res = []
    while not a3c.global_net.episode_reward_queue.empty():
        r = a3c.global_net.episode_reward_queue.get()
        res.append(r)
    plt.xlabel("情节数")
    plt.ylabel("奖励")
    plt.title("A3C算法解决Pendulum-v0问题")
    plt.plot(res)
    plt.show()


def show_chinese():
    from matplotlib import rcParams
    config = {
        "font.family": 'serif',
        "font.size": 14,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
        "axes.unicode_minus": False
    }
    rcParams.update(config)


if __name__ == "__main__":
    show_chinese()
    game_name = "Pendulum-v0"
    n_work = mp.cpu_count()
    envs = [gym.make(game_name) for _ in range(n_work)]
    lr = 1e-4
    discount = 0.9
    simple_a3c = SimpleA3C(lr, discount, envs)
    test_a3c(simple_a3c)
