'''
Description  : 
Author       : CagedBird
Date         : 2021-08-20 10:41:47
FilePath     : /rl/src/rl_algorithms/DDPG/test_ddpg.py
'''
from src.rl_algorithms.DDPG.simple_ddpg import SimpleDDPG
from src.utils.plot_episode_rewards import plot_episode_rewards
from time import time
import gym
import numpy as np


def test_ddpg(env: gym.Env, ddpg: SimpleDDPG, max_episodes=500):
    episode_reward_list = []
    mean_reward_list = []
    MEAN_INTERVAL = 50
    MAX_EPISODE_STEPS = 200

    for episode in range(max_episodes):
        observation = env.reset()
        episode_reward = 0
        for _ in range(MAX_EPISODE_STEPS):
            action = ddpg.choose_action(observation)
            action = ddpg.add_noise(action) # DDPG need add action noise
            next_observation, reward, done, _ = env.step(action)
            episode_reward += reward
            ddpg.learn(observation, action, reward / 10, next_observation)
            if done:
                break
            observation = next_observation
        """ if episode % 100 == 0:
            print("Episode: {}, reward: {}.".format(episode, episode_reward)) """
        # print("Episode: {}, reward: {}.".format(episode, episode_reward))
        episode_reward_list.append(episode_reward)
        mean_reward_list.append(sum(episode_reward_list[-MEAN_INTERVAL:]) / MEAN_INTERVAL)

    return episode_reward_list


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

def seed_all(seed, env=None):
    import os
    import numpy as np
    import random
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    seed_torch(seed)

def seed_torch(seed):
    """原文链接：https://blog.csdn.net/john_bh/article/details/107731443"""
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    show_chinese()
    env_name = "Pendulum-v0"
    my_env = gym.make(env_name)
    "seed the test"
    # seed = 1
    # seed_all(seed, my_env)
    "Parameters for ddpg"
    state_dim = my_env.observation_space.shape[0]  # 3个状态维度
    action_dim = my_env.action_space.shape[0]  # 1个动作维度
    action_bound = np.array([my_env.action_space.low, my_env.action_space.high])  # (2, 1)
    # hyper parameter
    lr_a = 1e-3
    lr_c = 2e-3
    discount = 0.9
    max_episodes = 500
    memory_capacity = 10000
    batch_size = 32
    tau = 1e-2
    noise_init_var = 3
    noise_decay = 0.9995
    simple_ddpg = SimpleDDPG(state_dim, action_dim, action_bound, lr_a, lr_c, discount, memory_capacity, batch_size, tau,
                             noise_init_var, noise_decay)
    tic = time()
    episode_reward_list = test_ddpg(my_env, simple_ddpg, max_episodes)
    toc = time()
    plot_episode_rewards(episode_reward_list, "DDPG", env_name, False, toc-tic)
    
