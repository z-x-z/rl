# coding=UTF-8
'''
Description  : 
Author       : caged_bird
Date         : 2021/1/22 9:03 PM
File         : rl - test_PG
'''
from simple_policy_gradients import SimplePolicyGradients
import matplotlib.pyplot as plt
import gym


def test_policy_gradients(env, policy_gradients: SimplePolicyGradients, MAX_EPISODES=400):
    total_step = 0
    episode_reward_list = []
    mean_reward_list = []
    MEAN_INTERVAL = 50

    for episode_i in range(MAX_EPISODES + 1):
        "1.初始化"
        observation = env.reset()
        episode_reward = 0
        while True:
            "2.在每个周期内利用蒙特卡洛法收集经验轨迹"
            action = policy_gradients.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            reward = -1 if done else reward  # 在结束时给予一定的惩罚
            episode_reward += reward

            policy_gradients.store_transition(observation, action, reward)
            observation = next_observation
            if done:
                break
        "3.结束一个episode后对整个经验轨迹进行学习"
        policy_gradients.learn()
        episode_reward_list.append(episode_reward)
        mean_reward_list.append(sum(episode_reward_list[-MEAN_INTERVAL:]) / MEAN_INTERVAL)

    plt.xlabel("情节数")
    plt.ylabel("奖励")
    plt.title("策略梯度网络算法解决CartPole-v0问题")
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
    simple_dqn = SimplePolicyGradients(n_observations, n_actions)
    test_policy_gradients(my_env, simple_dqn)

