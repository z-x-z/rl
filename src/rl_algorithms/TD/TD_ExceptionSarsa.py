import gym
import numpy as np
from collections import defaultdict, namedtuple
from TD import TDCartPole


class TD_CartPole_ExceptionSarsa(TDCartPole):
    def exceptionSarsa_core(self):
        nA = env.action_space.n
        Q = defaultdict(lambda: np.zeros(nA, dtype=np.float32))
        epsilon = 0.1
        record = namedtuple("Record", ["episode_steps", "episode_rewards"])
        record.episode_steps = []
        record.episode_rewards = []
        state_key_fun = self.get_bins_states
        "对每一个终止状态的所有动作价值函数设置为0"
        "终止状态："
        for episode in range(self.max_episodes):
            state = self.env.reset()
            "1. 在每一幕的开头，根据动作价值函数的epsilon贪婪法选择当前状态下的动作(sarsa专有)"
            action, _ = self.epsilon_greedy_policy(Q[state_key_fun(state)])
            total_step = 0
            total_reward = 0
            while (True):
                "1.1 执行一个单步操作"
                next_state, reward, done, _ = self.env.step(action)
                # 将价值所对应的键值转化为方便映射的键值
                state_key, next_state_key = state_key_fun(state), state_key_fun(next_state)
                "1.2 根据动作价值函数的epsilon贪婪法选择下一个状态下的动作(sarsa专有)"
                next_action, action_prob = self.epsilon_greedy_policy(Q[next_state_key])
                "1.3 单步更新动作价值函数"
                "下一个状态的收益值为下一个状态所有动作的动作概率乘以其动作价值(期望Sarsa独有)"
                next_gain = self.discount * sum([action_prob[a] * Q[next_state_key][a] for a in range(self.nA)]) + reward
                td_delta = next_gain - Q[state_key][action]
                Q[state_key][action] += self.learning_rate * td_delta
                # 进行调试记录
                total_step += 1
                total_reward += reward
                if (done):
                    break
                state = next_state
                action = next_action  # (sarsa专有)

            record.episode_steps.append(total_step)
            record.episode_rewards.append(total_reward)
        return Q, record

    def run(self):
        return self.exceptionSarsa_core()


def test_TD_CartPole_ExceptionSarsa():
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    exceptionSarsa = TD_CartPole_ExceptionSarsa(env, 200)
    action_values, record = exceptionSarsa.run()
    exceptionSarsa.draw_record(record)
