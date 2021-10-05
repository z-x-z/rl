import numpy as np
from src.environments.GridworldEnv import GridworldEnv


def policy_eval(environment, policy, discount=1.0, theta=1e-2):
    env = environment
    """ 策略评估算法 RLBook-p74-(4.4)
        input: policy (state_num, action_num), environment, discount, theta
        return: value_function (state_num)
    """
    V = np.zeros(env.nS, dtype=np.float32)  # 价值函数初始化 (16)
    "1.遍历所有状态"
    max_epoch = 50
    for epoch in range(max_epoch):
        delta = 0.  # delta用于每个epoch时进行比较        
        for state in range(env.nS):
            v = 0
            "2.遍历该策略下可以采取的动作"
            for action, action_prob in enumerate(policy[state]):
                "3.遍历在该(策略、动作)可到达的下一个状态"
                for state_prob, next_state, reward, _ in env.P[state][action]:
                    v += action_prob * state_prob * (reward + discount * V[next_state])

            delta = np.maximum(delta, np.abs(V[state] - v))
            V[state] = v

        if (delta <= theta):
            break
    return V  # (16)


def policy_improve(environment, V_pi, discount=1.):
    env = environment
    """ 策略提高 RLBook-p78,79-(4.6),(4.9)
        input: V_pi, environment
        return: improved_policy
    """
    improved_policy = np.zeros((env.nS, env.nA), dtype=np.float32)
    "1.对每个状态进行遍历"
    for state in range(env.nS):
        action_values = np.zeros(env.nA, dtype=np.float32)
        "2.对该状态下可能执行的动作进行遍历"
        for action in range(env.nA):
            "3.对（状态，动作），遍历所有的下一个转态"
            for state_prob, next_state, reward, done in env.P[state][action]:
                if (done and next_state != 15):  # 陷入到陷阱而非终点
                    action_values[action] = -np.inf
                    break
                else:
                    action_values[action] += state_prob * (reward + discount * V_pi[next_state])
        optimal_action = np.argmax(action_values)
        improved_policy[state][optimal_action] = 1.  # 设置最优策略中optimal_action索引为1，其他action为0

    return improved_policy


def policy_iteration(environment, _init_p_policy, discount=1.0):
    env = environment
    """
    策略迭代
        input: environment, init_policy, discount.
        return: optimized policy for current issue.
    """
    "1.初始化"
    p_policy = _init_p_policy.copy()
    max_episode = 10000
    "2.策略迭代核型"
    for episode in range(max_episode):
        "2.1策略评估"
        old_action = np.argmax(p_policy, axis=1)  # 确定性策略
        V_pi = policy_eval(env, p_policy, discount)
        "2.2策略提高"
        p_policy = policy_improve(env, V_pi, discount)
        optimal_action = np.argmax(p_policy, axis=1)
        # ? 打印调试信息
        print("Episode%d: " % (episode))
        print("V_pi:\n",V_pi.reshape(env.shape))
        print("Policy action:\n" ,optimal_action.reshape(env.shape), "\n")
        "2.3评判策略是否达到稳定"
        if (old_action == optimal_action).all():  # 对np而言all表示与，any表示或
            break

    return p_policy


if __name__ == "__main__":
    """
    [[1 1 2 3]
    [2 0 2 0]
    [1 1 2 0]
    [0 1 1 0]]"""
    env = GridworldEnv()  # 初始化环境变量
    init_p_policy = np.ones((env.nS, env.nA), dtype=np.float32) / 4  # 初始化策略
    optimal_p_policy = policy_iteration(env, init_p_policy, discount=0.8)  # 策略迭代
    print("optimal_policy:\n", np.argmax(optimal_p_policy, axis=1).reshape(env.desc.shape))