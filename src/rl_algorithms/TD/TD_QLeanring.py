import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, namedtuple
from TD import TDCartPole


class TDCartPoleQLearning(TDCartPole):
    def q_learning_core(self):
        """
            QLearning: 离轨策略
                动作选择策略：Epsilon greedy
                Q表更新策略：Max greedy
        """
        Q = defaultdict(lambda: np.zeros(self.nA, dtype=np.float32))
        record = {"episode_rewards": [], "episode_steps": [], "episode_mean_steps": []}
        state_key_fun = self.get_bins_states
        "对每一个终止状态的所有动作价值函数设置为0"
        "终止状态："
        for episode in range(self.max_episodes):
            state = self.env.reset()
            total_step = 0
            total_reward = 0
            while True:
                "1.1 在某一幕的每一步中，根据动作价值函数Q，利用epsilon贪婪选择一个动作(QLearning独有)"
                state_key = state_key_fun(state)
                action, _ = self.epsilon_greedy_policy(Q[state_key])
                "1.2 执行一个单步操作"
                next_state, reward, done, _ = self.env.step(action)
                # 将价值所对应的键值转化为方便映射的键值
                next_state_key = state_key_fun(next_state)
                "1.4 单步更新动作价值函数"
                "1.4.1 从下一个状态的动作价值函数中选择使其达到最大的动作(QLeanring独有)"
                next_optimal_action = np.argmax(Q[next_state_key])
                next_gain = reward + self.discount * Q[next_state_key][next_optimal_action]
                td_delta = next_gain - Q[state_key][action]
                Q[state_key][action] += self.learning_rate * td_delta
                # 进行调试记录
                total_step += 1
                total_reward += reward
                if done:
                    break
                state = next_state

            record["episode_steps"].append(total_step)
            record["episode_rewards"].append(total_reward)
            record["episode_mean_steps"].append(
                    sum(record["episode_steps"][-self.moving_average_interval:]) / self.moving_average_interval)
        return Q, record

    def run(self):
        return self.q_learning_core()


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    episode_nums = 200
    q_learning = TDCartPoleQLearning(env, episode_nums)
    Q, record = q_learning.run()
    # plot record
    TDCartPole.draw_record(record)
