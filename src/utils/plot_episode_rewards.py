'''
Description  : 
Author       : CagedBird
Date         : 2021-10-10 20:46:13
FilePath     : /rl/src/utils/plot_episode_reward.py
'''
import matplotlib.pyplot as plt
from src.utils.show_chinese import show_chinese
import platform

def plot_episode_rewards(episode_rewards, algorithm_name, env_name, use_gpu, second_cost=-1):
    show_chinese()
    plt.xlabel("情节数")
    plt.ylabel("奖励")
    plt.title("{}算法解决{}问题\n({}-{}, 耗时{:3f}s)".format(
        algorithm_name, env_name,
        platform.system(), "gpu" if use_gpu else "cpu", 
        second_cost))
    plt.plot(episode_rewards, label="每一情节下的奖励")
    plt.legend()
    plt.show()