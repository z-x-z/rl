import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, namedtuple
from src.rl_algorithms.TD.TD import TDCartPole


# This is comment
class TD_CartPole_DoubleQLearning(TDCartPole):
    def dqLeanring_core(self):
        Q1 = defaultdict(lambda: np.zeros(self.nA, dtype=np.float32))
        Q2 = defaultdict(lambda: np.zeros(self.nA, dtype=np.float32))
        record = namedtuple("Record", ["episode_steps", "episode_rewards"])
        record.episode_steps = []
        record.episode_rewards = []
        state_key_fun = self.get_bins_states
        "对每一个终止状态的所有动作价值函数设置为0"
        "终止状态："
        for episode in range(self.max_episodes):
            state = self.env.reset()
            total_step = 0
            total_reward = 0
            while (True):
                "1.1 在某一幕的每一步中"
                state_key = state_key_fun(state)
                "根据动作价值函数Q1、Q2，利用epsilon贪婪选择一个动作"
                action, _ = self.epsilon_greedy_policy(Q1[state_key] + Q2[state_key])
                "1.2 执行一个单步操作"
                next_state, reward, done, _ = self.env.step(action)
                # 将价值所对应的键值转化为方便映射的键值
                next_state_key = state_key_fun(next_state)
                "1.4 单步更新动作价值函数"
                "1.4.1 从下一个状态的动作价值函数中选择使其达到最大的动作"
                q1, q2 = (Q1, Q2) if np.random.random() > 0.5 else (Q2, Q1)
                next_optimal_action = np.argmax(q1[next_state_key])
                "discount所乘的是q2"
                next_gain = reward + self.discount * q2[next_state_key][next_optimal_action]
                td_delta = next_gain - q1[state_key][action]
                q1[state_key][action] += self.learning_rate * td_delta
                # 进行调试记录
                total_step += 1
                total_reward += reward
                if done:
                    break
                state = next_state

            record.episode_steps.append(total_step)
            record.episode_rewards.append(total_reward)
        return Q1, Q2, record

    def run(self):
        return self.dqLeanring_core()


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    dqLearning = TD_CartPole_DoubleQLearning(env, max_episodes=200)
    Q1, Q2, record = dqLearning.run()
    # plot record
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(record.episode_steps[:200])
    plt.show()

# To experience a pillow