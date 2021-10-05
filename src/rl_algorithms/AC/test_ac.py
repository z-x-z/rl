'''
Description  : 
Author       : CagedBird
Date         : 2021-08-13 15:19:51
FilePath     : /rl/src/rl_algorithms/AC/test_a2c.py
'''

# from cagedbird_rl.mofan_rl.DQN.simple_dqn import SimpleDeepQNetwork
from numpy.lib.function_base import select
from src.rl_algorithms.AC.simple_ac import SimpleAC
import matplotlib.pyplot as plt
import gym


def test_ac(env: gym.Env, ac: SimpleAC, MAX_EPISODES=500):
    episode_reward_list = []
    mean_reward_list = []    
    MEAN_INTERVAL = 50
    MAX_EPISODE_STEPS = 300

    for _ in range(MAX_EPISODES + 1):
        observation = env.reset()
        episode_reward = 0
        for _ in range(MAX_EPISODE_STEPS):
            action = ac.choose_action(observation)
            next_observation, reward, done, _ = env.step(action)
            reward = -20 if done else reward  # 在结束时给予一定的惩罚
            observation = next_observation
            episode_reward += reward
            ac.learn(observation, next_observation, action, reward)
            if done:
                break

        episode_reward_list.append(episode_reward)
        mean_reward_list.append(sum(episode_reward_list[-MEAN_INTERVAL:]) / MEAN_INTERVAL)

    # plt.plot(list(range(len(episode_step_list))), episode_step_list)
    plt.xlabel("情节数")
    plt.ylabel("奖励")
    plt.title("AC算法解决CartPole-v0问题")
    plt.plot(episode_reward_list, label="每一情节下的奖励")
    plt.plot(mean_reward_list, label="平均奖励")
    plt.legend()
    plt.show()

def show_chinese():
    from matplotlib import rcParams
    config = {
        "font.family":'serif',
        "font.size": 14,
        "mathtext.fontset":'stix',
        "font.serif": ['SimSun'],
        "axes.unicode_minus": False
    }
    rcParams.update(config)

if __name__ == "__main__":
    show_chinese()
    my_env = gym.make("CartPole-v0")
    n_observations = my_env.observation_space.shape[0]  # 4个状态量
    n_actions = my_env.action_space.n  # 2种动作
    lr_a = 1e-3
    lr_c = 1e-2
    discount = 0.95
    simple_ac = SimpleAC(n_observations, n_actions, lr_a, lr_c, discount)
    test_ac(my_env, simple_ac, 1000)
