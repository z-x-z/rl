# from cagedbird_rl.mofan_rl.DQN.simple_dqn import SimpleDeepQNetwork
from simple_dqn import SimpleDeepQNetwork
from src.utils.show_chinese import show_chinese
import time
import platform
import matplotlib.pyplot as plt
import gym


def test_dqn(env, dqn: SimpleDeepQNetwork, MAX_EPISODES=400, START_LEARNING_STEPS=200, LEARNING_INTERVAL=5):
    tic = time.time()
    total_step = 0
    episode_reward_list = []
    mean_reward_list = []
    MEAN_INTERVAL = 50
    is_train = True

    for episode_i in range(MAX_EPISODES + 1):
        if episode_i > int(MAX_EPISODES * 0.8):
            is_train = False
        observation = env.reset()
        episode_reward = 0

        while True:
            action = dqn.choose_action(observation, is_train)
            next_observation, reward, done, info = env.step(action)
            reward = -1 if done else reward  # 在结束时给予一定的惩罚
            dqn.store_transition(observation, action, reward, next_observation)
            total_step += 1
            # 当总步长大于200时，每5个步长学习一次
            if total_step > START_LEARNING_STEPS and total_step % LEARNING_INTERVAL == 0:
                dqn.learn()

            observation = next_observation
            episode_reward += reward
            if done:
                break
        episode_reward_list.append(episode_reward)
        mean_reward_list.append(sum(episode_reward_list[-MEAN_INTERVAL:]) / MEAN_INTERVAL)

    toc = time.time()
    time_cost = toc - tic
    # plt.plot(list(range(len(episode_step_list))), episode_step_list)
    plt.xlabel("情节数")
    plt.ylabel("奖励")
    plt.title("DQN算法解决CartPole-v0问题\n({}-{}, 耗时{:3f}s)".format(platform.system(), "gpu" if dqn.use_gpu else "cpu", time_cost))
    plt.plot(episode_reward_list, label="每一情节下的奖励")
    plt.plot(mean_reward_list, label="平均奖励")
    plt.legend()
    plt.show()
    # plt.savefig("figure.png")




if __name__ == "__main__":
    show_chinese()
    my_env = gym.make("CartPole-v0")
    n_observations = my_env.observation_space.shape[0]  # 4个状态量
    n_actions = my_env.action_space.n  # 2种动作
    use_gpu = False
    simple_dqn = SimpleDeepQNetwork(n_observations, n_actions, e_greedy_increment=0.9 / 200, use_gpu=use_gpu)
    test_dqn(my_env, simple_dqn, 2000)
