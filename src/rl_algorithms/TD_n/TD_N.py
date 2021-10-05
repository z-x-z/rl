import numpy as np
from numpy.lib.function_base import select
import pandas as pd
import matplotlib.pyplot as plt


class TD_N:
    def __init__(self, n, env, max_episodes, learning_rate=0.5, discount=1.0, epsilon=0.1, moving_average_interval=20):
        self.n = n
        self.env = env
        self.nA = env.action_space.n
        self.max_episodes = max_episodes
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.moving_average_interval = moving_average_interval

    def run(self):
        """算法的核心调用方法"""
        pass


class TD_N_CartPole(TD_N):
    def __init__(self, n, env, max_episodes, learning_rate=0.5, discount=1.0, epsilon=0.1, n_bins=10):
        super().__init__(n, env, max_episodes, learning_rate, discount, epsilon)
        self.cart_position_bins = pd.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1]
        self.pole_angle_bins = pd.cut([-2, 2], bins=n_bins, retbins=True)[1]
        self.cart_velocity_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1]
        self.angle_rate_bins = pd.cut([-3.5, 3.5], bins=n_bins, retbins=True)[1]

    def get_bins_states(self, state):
        """ 将连续（实数列表）空间的状态转化到离散（整形元祖）空间，结果可作为字典的键值
        """
        s1_, s2_, s3_, s4_ = state
        cart_position_idx = np.digitize(s1_, self.cart_position_bins)
        pole_angle_idx = np.digitize(s2_, self.pole_angle_bins)
        cart_velocity_idx = np.digitize(s3_, self.cart_velocity_bins)
        angle_rate_idx = np.digitize(s4_, self.angle_rate_bins)

        state_ = [cart_position_idx, pole_angle_idx, cart_velocity_idx, angle_rate_idx]
        return tuple(map(lambda s: int(s), state_))

    def epsilon_greedy_policy(self, state_q):
        action_prob = np.ones(self.nA, dtype=np.float32) * self.epsilon / self.nA
        action_prob[np.argmax(state_q)] = 1 - self.epsilon + self.epsilon / self.nA
        action = np.random.choice(np.arange(self.nA), p=action_prob)
        return action, action_prob

    def next_action(self, state_q, next_action_=None):
        if not next_action_:
            next_action_, _ = self.epsilon_greedy_policy(state_q)
            return next_action_

    @staticmethod
    def draw_record(record, TD_type_str=None):
        # Plot the episode length over time
        plt.figure(figsize=(10, 5))
        plt.plot(record["episode_steps"])
        plt.xlabel("Episode")
        plt.ylabel("Episode steps")
        plt.title("Episode steps over Time ({})".format(TD_type_str) if TD_type_str else "")

        # Plot the episode reward over time
        plt.figure(figsize=(10, 5))
        plt.plot(record["episode_mean_steps"])
        plt.xlabel("Episode")
        plt.ylabel("Episode mean reward")
        plt.title("Episode mean reward over Time. ({})".format(TD_type_str) if TD_type_str else "")
        plt.show()
