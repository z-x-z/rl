from src.environments.GridworldEnv import GridworldEnv
import numpy as np


def get_action_values(environment, state, state_values, discount=1.0):
    "对某一状态而言的动作价值函数"
    env = environment
    "初始化动作价值函数"
    action_values = np.zeros((env.nA), dtype=np.float32)
    for action in range(env.nA):
        for prob, next_state, reward, done in env.P[state][action]:
            action_values[action] += prob * (reward + discount * state_values[next_state])

    return action_values  # (env.nA)


def value_iteration(environment, theta=1e-1, discount=1.0):
    env = environment
    "初始化状态价值函数"
    state_values = np.zeros((env.nS), dtype=np.float32)
    max_episodes = 50

    "值迭代核心算法"
    for episode in range(max_episodes):
        delta = 0.
        "对每一状态迭代"
        for state in range(env.nS):
            action_values = get_action_values(env, state, state_values, discount)  # (nA)
            optimal_action_values = np.max(action_values)
            delta = np.maximum(delta, np.abs(optimal_action_values - state_values[state]))
            state_values[state] = optimal_action_values
        # print("episode:", episode, "state_values", state_values, "\n")
        if (delta <= theta):
            break

    "策略评估"
    optimal_p_policy = np.zeros((env.nS, env.nA), dtype=np.float32)
    for state in range(env.nS):
        action_values = get_action_values(env, state, state_values, discount)  # (nA)
        optimal_action = np.argmax(action_values)
        optimal_p_policy[state, optimal_action] = 1.
    return optimal_p_policy, state_values


if __name__ == "__main__":
    env = GridworldEnv()
    p_policy, state_values = value_iteration(env, discount=0.9)  # (nS, nA), (nS)
    # np.argmax(p_policy, axis=1): 将概率策略用贪心算法转化为确定策略
    print(np.argmax(p_policy, axis=1).reshape(env.desc.shape))
    print(state_values.reshape(env.desc.shape))