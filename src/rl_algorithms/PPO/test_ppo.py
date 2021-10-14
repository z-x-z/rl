'''
Description  : 
Author       : CagedBird
Date         : 2021-09-25 23:01:37
FilePath     : /rl/src/rl_algorithms/PPO/test_ppo.py
'''
from time import time
from src.utils.plot_episode_rewards import plot_episode_rewards
import gym
from src.rl_algorithms.PPO.simple_ppo import SimplePPO


def test_ppo(env: gym.Env, ppo: SimplePPO, MAX_EPISODES=1000, MAX_EPISODE_STEPS=200):
    episode_rewards = []
    for e in range(MAX_EPISODES):
        s = env.reset()
        episode_reward = 0
        for step in range(MAX_EPISODE_STEPS):
            a, log_prob = ppo.choose_action(s)
            s_, r, done, _ = env.step(a)
            episode_reward += r
            ppo.buffer_push(s, a, r, log_prob)
            s = s_
            if (step + 1) % ppo.BATCH_SIZE == 0 or step == (MAX_EPISODE_STEPS - 1) or done:
                ppo.learn(s_)
                if done:
                    break
        episode_rewards.append(episode_reward)
    return episode_rewards


if __name__ == "__main__":
    env_name = "Pendulum-v0"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ppo = SimplePPO(state_dim, action_dim)
    tic = time()
    episode_rewards = test_ppo(env, ppo)
    toc = time()
    plot_episode_rewards(episode_rewards, "PPO", env_name, False, toc-tic)
