'''
Description  : 
Author       : CagedBird
Date         : 2020-12-14 22:11:22
FilePath     : /rl/cagedbird_rl/my_codes/MC/mc_prediction.py
'''

import numpy as np
import gym
from plot_value_function import plot_value_function
from collections import defaultdict


def simple_policy(state):
    "简单策略：状态到动作的映射"
    player_score, _, _ = state
    return 0 if player_score >= 18 else 1


def mc_trajectory_collect(episode_trajectory, values_sum, values_count, discount=1.0, is_first=True):
    trajectory_len = len(episode_trajectory)
    gains = np.zeros(trajectory_len + 1, dtype=np.float32)
    "一遍计算当前经验轨迹所有时刻的收益值: gain[i] = reward + discount * gain[i+1]"
    for i, data_i in enumerate(reversed(episode_trajectory)):
        i = trajectory_len - i - 1  # i = 0时，实际上i对应的下标为len-i-1
        gains[i] = data_i[2] + discount * gains[i + 1]  # data_i: (s, a, r)
    "收集"
    if (is_first):
        "首次访问"
        state_set = set()
        for i, data_i in enumerate(episode_trajectory):
            state = data_i[0]
            if (state in state_set):
                continue
            state_set.add(state)
            values_sum[state] += gains[i]
            values_count[state] += 1.0
    else:
        "每次访问"
        for i, data_i in enumerate(episode_trajectory):
            state = data_i[0]
            values_sum[state] += gains[i]
            values_count[state] += 1.0


def mc_firstvisit_prediction(environment, policy, max_episodes=10000, episode_endtime=10, discount=1.0):
    """ 蒙特卡洛-首次访问预测算法
        input: environment, policy, MAX_EPISODES, episode_endtime, discount
        return: state_values
    """
    "1.初始化"
    env = environment
    state_values = defaultdict(float)  # 状态函数
    values_sum = defaultdict(float)  # 状态函数在所有幕的首次总和
    values_count = defaultdict(float)  # 状态函数在所有幕的首次出现次数之和
    "2.蒙特卡洛预测"
    "2.1 对每一幕有"
    for episode in range(max_episodes):
        episode_trajectory = []
        state = env.reset()
        "2.2 访问每一幕的一条支链直到达到终点（或达到endtime）"
        for i in range(episode_endtime):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_trajectory.append((state, action, reward))
            if (done):
                break
            state = next_state
        "2.3 进行蒙特卡洛经验轨迹数据的收集"
        mc_trajectory_collect(episode_trajectory, values_sum, values_count, discount, is_first=True)

    "3.对收集的值函数进行平均化"
    for state in values_count:
        state_values[state] = values_sum[state] / values_count[state]
    return state_values


def mc_everyvisit_prediction(environment, policy, max_episodes=10000, episode_endtime=10, discount=1.0):
    """ 蒙特卡洛-每次访问预测算法
        input: environment, policy, MAX_EPISODES, episode_endtime, discount
        return: state_values
    """
    "1.初始化"
    env = environment
    state_values = defaultdict(float)  # 状态函数
    values_sum = defaultdict(float)  # 状态函数在所有幕的首次总和
    values_count = defaultdict(float)  # 状态函数在所有幕的首次出现次数之和
    "2.蒙特卡洛预测"
    "2.1 对每一幕有"
    for episode in range(max_episodes):
        episode_trajectory = []
        state = env.reset()
        "2.2 访问每一幕的一条支链直到达到终点（或达到endtime）"
        for i in range(episode_endtime):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_trajectory.append((state, action, reward))
            if (done):
                break
            state = next_state
        "2.3 进行蒙特卡洛经验轨迹数据的收集"
        mc_trajectory_collect(episode_trajectory, values_sum, values_count, discount, is_first=False)

    "3.对收集的值函数进行平均化"
    for state in values_count:
        state_values[state] = values_sum[state] / values_count[state]
    return state_values

if __name__ == "__main__":
    env = gym.make("Blackjack-v0")
    # v2 = mc_everyvisit_prediction(env, simple_policy, 1000)
    v2 = mc_firstvisit_prediction(env, simple_policy, 1000)
    plot_value_function(v2, "MC every visit")
    for key in sorted(v2):
        print(key, v2[key])