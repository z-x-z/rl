'''
Description  : 
Author       : CagedBird
Date         : 2021-09-25 23:01:37
FilePath     : /rl/src/rl_algorithms/PPO/test_ppo.py
'''
from _typeshed import OpenBinaryMode
import gym
from src.rl_algorithms.PPO.simple_ppo import SimplePPO


def test_ppo(env: gym.Env, ppo: SimplePPO, MAX_EPISODES=400):
    MAX_EPISODE_STEPS = 32
    BUFFER_SIZE = 8
    discount = 0.9
    for e in range(MAX_EPISODES):
        s = env.reset()
        # TODO buffer initialization
        for step in range(MAX_EPISODE_STEPS):
            a = ppo.choose_action(s)
            s_, r, done, _ = env.step(a)
            # TODO buffer collect (s, a, r)
            s = s_
            if (step+1) % BUFFER_SIZE == 0 or step == (MAX_EPISODE_STEPS-1):
                bs, ba, br = None
                # TODO get (bs, ba, br) from buffer
                v_s_ = ppo.get_state_value(s_)
                bg = []
                for r in br:
                    v_s_ = r + discount * v_s_
                    bg.append(v_s_)
                bg.reverse()
                ppo.learn(bs, ba, bg)
