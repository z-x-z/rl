'''
Description  : 
Author       : CagedBird
Date         : 2020-12-14 22:07:00
LastEditors  : Please set LastEditors
LastEditTime : 2021-06-12 22:36:29
FilePath     : /rl/cagedbird_rl/my_codes/DP/dp_value_iteration.py
'''
from src.environments.GridworldEnv import GridworldEnv
import numpy as np


def get_action_values(env, state, state_values, discount=1.0):
    """对某一状态而言的动作价值函数
    """
    "初始化动作价值函数"
    action_values = np.zeros((env.nA), dtype=np.float32)
    for action in range(env.nA):
        for prob, next_state, reward, done in env.P[state][action]:
            action_values[action] += prob * (reward + discount * state_values[next_state])

    return action_values  # (env.nA)


def dp_episode_value_iteration(env, state_values, theta=1e-3, discount=1.0):
    delta = 0.
    "2.1.对每一状态迭代"
    for state in range(env.nS):
        action_values = get_action_values(env, state, state_values, discount)  # (nA)
        optimal_action_values = np.max(action_values)
        delta = np.maximum(delta, np.abs(optimal_action_values - state_values[state]))
        state_values[state] = optimal_action_values
    return delta


def dp_policy_evaluation(env, state_values, discount):
    optimal_p_policy = np.zeros((env.nS, env.nA), dtype=np.float32)
    for state in range(env.nS):
        action_values = get_action_values(env, state, state_values, discount)  # (nA)
        optimal_action = np.argmax(action_values)
        optimal_p_policy[state, optimal_action] = 1.  # 确定性策略
    return optimal_p_policy


def dp_value_iteration(env, theta=1e-3, discount=1.0):
    "1.初始化状态价值函数"
    state_values = np.zeros((env.nS), dtype=np.float32)
    max_episodes = 50
    "2.值迭代核心算法"
    for episode in range(max_episodes):
        delta = dp_episode_value_iteration(env, state_values, theta, discount)
        # print("episode:", episode, "state_values:\n", state_values.reshape(env.shape), "\n")
        if (delta <= theta):
            break
    "3.策略评估"
    optimal_p_policy = dp_policy_evaluation(env, state_values, discount)
    return optimal_p_policy, state_values


if __name__ == "__main__":
    env = GridworldEnv()
    # p_policy 概率策略
    optimal_p_policy, state_values = dp_value_iteration(env)
    print("policy:\n", np.argmax(optimal_p_policy, axis=1).reshape(env.desc.shape))
    print("state:\n", state_values)
