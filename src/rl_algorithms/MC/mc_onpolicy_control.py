'''
Description  : 
Author       : CagedBird
Date         : 2020-12-14 22:11:22
LastEditors  : Please set LastEditors
LastEditTime : 2021-06-13 16:17:36
FilePath     : /rl/cagedbird_rl/my_codes/MC/mc_onpolicy_control.py
'''
import gym
import numpy as np
from collections import defaultdict
from plot_value_function import plot_value_function


def epsilon_greedy_policy(state_action_values, epsilon=0.1):
    """
        input: state_action_values (np(nA)), state, epsilon
        return: policy (np(nA))
    """
    nA = state_action_values.shape[0]
    policy = np.ones((nA), dtype=np.float32) * (epsilon / nA)
    optimal_action = np.argmax(state_action_values)
    policy[optimal_action] = 1 - epsilon + epsilon / nA
    return policy


def mc_control_trajectory_collect(episode_trajectory, values_sum, values_count, discount=1.0, is_first=True):
    trajectory_len = len(episode_trajectory)
    gains = np.zeros(trajectory_len + 1, dtype=np.float32)
    "1.一遍计算当前经验轨迹所有时刻的收益值: gain[i] = reward + discount * gain[i+1]"
    for i, data_i in enumerate(reversed(episode_trajectory)):  # data_i : ((s, a), r)
        i = trajectory_len - i - 1  # i = 0时，实际上i对应的下标为len-i-1
        gains[i] = data_i[1] + discount * gains[i + 1]
    "2.收集"
    if (is_first):
        "首次访问"
        sa_set = set()
        for i, data_i in enumerate(episode_trajectory):
            sa_pair = data_i[0]
            if (sa_pair in sa_set):
                continue
            sa_set.add(sa_pair[0])
            values_sum[sa_pair] += gains[i]
            values_count[sa_pair] += 1.0
    else:
        "每次访问"
        for i, data_i in enumerate(episode_trajectory):
            sa_pair = data_i[0]
            values_sum[sa_pair] += gains[i]
            values_count[sa_pair] += 1.0


def mc_everyvisit_epsilon_greedy_control(environment, max_episodes=10000, epsilon=0.1, episode_max_steps=10, discount=1.0):
    """蒙特卡洛-每次访问-epsilon贪婪控制算法
        input: environment, epsilon=0.1, MAX_EPISODES=100, episode_endtime=10, discount=1.0
        return: action_values (nS, nA)
    """
    env = environment
    trajectory_collect = mc_control_trajectory_collect
    nA = env.action_space.n
    # 每个动作价值函数默认为一个np数组（长度为nA）
    action_values = defaultdict(lambda: np.zeros(nA))
    values_sum = defaultdict(float)
    values_count = defaultdict(float)

    "对每一幕而言"
    for episode in range(max_episodes):
        episode_trajectory = []
        state = env.reset()
        "1.使用策略生成经验轨迹"
        for i in range(episode_max_steps):
            "1.1按照epsilon贪婪选择当前的动作"
            action_prob = epsilon_greedy_policy(action_values[state], epsilon)
            action = np.random.choice(np.arange(action_prob.shape[0]), p=action_prob)  # 按照相应的概率选择元素
            "1.2完成一个动作"
            next_state, reward, done, _ = env.step(action)
            episode_trajectory.append(((state, action), reward))  # ((s ,a), r)
            if (done):
                break
            state = next_state

        "2.使用每次访问算法收集经验轨迹中的数据"
        trajectory_collect(episode_trajectory, values_sum, values_count, discount, is_first=False)
        "3.平均化处理，并赋值到动作价值函数。（每一幕结束前都要重新赋值，因为在之后的一幕中会用到更新后的动作价值函数）"
        for sa_pair in values_sum:
            action_values[sa_pair[0]][sa_pair[1]] = values_sum[sa_pair] / values_count[sa_pair]

    return action_values


if __name__ == "__main__":
    env = gym.make("Blackjack-v0")
    action_values = mc_everyvisit_epsilon_greedy_control(env, discount=0.8)
    v = defaultdict(float)
    for state, values in action_values.items():
        v[state] = np.max(values)
    plot_value_function(v, title="Optimal Value Function")