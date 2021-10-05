import gym
import numpy as np
from collections import defaultdict
from plot_value_function import plot_value_function

def random_policy(nA):
    "随机策略"

    def policy(state):
        action_prob = np.ones((nA), dtype=np.float32) / nA  # 每个动作的选择概率相同，故随机选择所有的动作
        return action_prob

    return policy


def greedy_policy(action_values):
    "贪婪策略"

    def policy(state):
        optimal_action = np.argmax(action_values[state])
        action_prob = np.zeros_like(action_values[state], dtype=np.float32)
        action_prob[optimal_action] = 1.0
        return action_prob

    return policy


def mc_weightedimportance_everyvisit_control(environment, max_episodes=10000, episode_max_steps=20, discount=1.0):
    """
        input: environment, MAX_EPISODES, episode_endtime, discount
        return: action_values
    """
    env = environment
    nA = env.action_space.n
    importance_weight_sum = defaultdict(lambda: np.zeros((nA), np.float32))
    action_values = defaultdict(lambda: np.zeros((nA), np.float32))

    "行为策略为随机选策略，目标策略为贪婪策略"
    behaviour_policy = random_policy(nA)  # 行为策略选择随机策略
    target_policy = greedy_policy(action_values)  # 目标策略选择贪婪策略

    "对每一幕进行采样"
    for episode in range(max_episodes):
        state = env.reset()
        trajectory = []
        "1.收集一条经验轨迹"
        for t in range(episode_max_steps):
            action_prob = behaviour_policy(state)  # 根据行为策略获取动作选择概率
            action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, action, reward))
            if (done):
                break
            state = next_state
        "2.根据经验轨迹进行加权平均"
        gain = 0
        w = 1
        Q, C = action_values, importance_weight_sum  # 别名
        for (state, action, reward) in trajectory[::-1]:
            gain = discount * gain + reward
            C[state][action] += w
            Q[state][action] += w / C[state][action] * (gain - Q[state][action])
            if (action != np.argmax(target_policy(state))):  # 若当前的动作不等于目标策略所选的动作，则停止本次加权平均
                break
            w = w / behaviour_policy(state)[action]

    return action_values


def test_mc_weightedimportance_everyvisit_control():
    env = gym.make("Blackjack-v0")
    action_values = mc_weightedimportance_everyvisit_control(env, 1000)
    v = defaultdict(float)
    for state, values in action_values.items():
        v[state] = np.max(values)
    for state in sorted(v):
        print(state, v[state])


if __name__ == "__main__":
    env = gym.make("Blackjack-v0")
    action_values = mc_weightedimportance_everyvisit_control(env, 10000, discount=0.8)
    v = defaultdict(float)
    for state, values in action_values.items():
        v[state] = np.max(values)
    plot_value_function(v, title="Optimal Value Function")