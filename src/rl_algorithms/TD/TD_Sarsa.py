from cagedbird_rl.my_codes.TD.TD import TD
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, namedtuple
from TD import TDCartPole


class TDCartPoleSarsa(TDCartPole):
    def sarsa_core(self):
        """
                Sarsa: 同轨策略
                    动作选择策略: Epsilon greedy
                    Q表更新策略: Epsilon greedy
                    动作选择策略 == Q表更新策略
        """
        nA = self.env.action_space.n  # 2
        Q = defaultdict(lambda: np.zeros(nA, dtype=np.float32))
        record = {"episode_rewards": [], "episode_steps": [], "episode_mean_steps": []}
        state_key_fun = self.get_bins_states
        "对每一个终止状态的所有动作价值函数设置为0"
        "终止状态："
        for episode in range(self.max_episodes):
            state = self.env.reset()
            "1. 在每一幕的开头，根据动作价值函数的epsilon贪婪法选择当前状态下的动作(sarsa专有)"
            action, _ = self.epsilon_greedy_policy(Q[state_key_fun(state)])
            total_step = 0
            total_reward = 0
            while True:
                "1.1 执行一个单步操作"
                next_state, reward, done, _ = self.env.step(action)
                # 将价值所对应的键值转化为方便映射的键值
                state_key, next_state_key = state_key_fun(state), state_key_fun(next_state)
                "1.2 根据动作价值函数的epsilon贪婪法选择下一个状态下的动作(sarsa专有)"
                next_action, _ = self.epsilon_greedy_policy(Q[next_state_key])  # 动作选择策略：Epsilon greedy
                "1.3 单步更新动作价值函数"
                next_gain = reward + self.discount * Q[next_state_key][next_action]  # Q表更新策略:  Epsilon greedy
                td_delta = next_gain - Q[state_key][action]
                Q[state_key][action] += self.learning_rate * td_delta
                # 进行调试记录
                total_step += 1
                total_reward += reward
                if done:
                    break
                state = next_state
                action = next_action  # (sarsa专有)

            record["episode_rewards"].append(total_reward)
            record["episode_steps"].append(total_step)
            record["episode_mean_steps"].append(
                    sum(record["episode_steps"][-self.moving_average_interval:]) / self.moving_average_interval)

        return Q, record

    def run(self):
        return self.sarsa_core()


if __name__ =="__main__":
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    episode_nums = 200
    sarsa = TDCartPoleSarsa(env, episode_nums)
    Q, record = sarsa.run()
    TDCartPole.draw_record(record)
