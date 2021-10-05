# coding: utf-8

import numpy as np
import sys
from time import sleep

from gym import utils
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
ACTION_MAP = ["up", "right", "down", "left"]
ARROW_MAP = ["↑", "→", "↓", "←"]
MAPS = {
    '5*5': ["OOOOO", "OSOOO", "OOOXO", "OOXGO", "OOOOO"],
    '4*4': ['OOOS', 'XOOX', 'OXOO', 'OOGO'],
    "6*6": ['OXOOXO', 'OGOXXO', 'OXXXOO', 'OOOOOO', 'OXOOXO', 'OOOOXS'],
    "8*6": ['OOOOSO', 'OOOXXX', 'OOOOOO', 'OXOOOO', 'OXOOOO', 'OOXOXX', 'XXOOOO', 'OXOXGO'],
    "10*10": [
        'OOOOOOOOOO', 'SXOOOOXXXO', 'OOOOOXOOXO', 'OOOXOOXXOX', 'OOXOXOXOOO',\
        'OXOXOXXOXX', 'OOOOOXOOOO', 'OXXXOOOOOO', 'XXXOXOXOOO', 'OOXOXOOOOG']
}
REWARD_MAP = {b'O': 0, b'S': 0, b'X': -1, b'G': 1}


class GridworldEnv(discrete.DiscreteEnv):
    """
    FrozenLakeEnv1 is a copy environment from GYM toy_text FrozenLake-01

    You are an agent on an 4x4 grid and your goal is to reach the terminal
    state at the bottom right corner.

    For example, a 4x4 grid looks as follows:

    S  O  O  O
    O  X  O  X
    O  O  O  X
    X  O  O  G

    S : starting point, safe
    O : frozen surface, safe
    X : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.

    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.

    step()
        return next_state, reward, done, {'prob': p}
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name='4*4'):
        self.desc = desc = np.asarray(MAPS[map_name], dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.shape = desc.shape

        nA = 4  # 动作集个数
        nS = np.prod(desc.shape)  # 状态集个数 = desc.shape[0] * desc.shape[1] * ...

        MAP_ROWS = desc.shape[0]
        MAP_COLS = desc.shape[1]
        MAP_CELLS = MAP_ROWS * MAP_COLS
        # Differ state nums for action.
        ACTIONS_DS = [-MAP_COLS, 1, MAP_COLS, -1]

        # initial state distribution [ 1.  0.  0.  ...]
        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {}
        state_grid = np.arange(nS).reshape(self.shape)
        it = np.nditer(state_grid, flags=['multi_index'])

        def is_done(s):
            if (s < 0 or s >= MAP_CELLS):
                return True
            else:
                # / 单纯的出发返回的是浮点数，//返回的则是整数
                return desc[s // MAP_COLS][s % MAP_COLS] in b'GX'

        def get_reward(s):
            if (s < 0 or s >= MAP_CELLS):
                return -1
            else:
                return REWARD_MAP[desc[s // MAP_COLS][s % MAP_COLS]]

        while not it.finished:
            s = it.iterindex
            r, c = it.multi_index
            # P[s][a] == [(probability, nextstate, reward, done), ...]
            P[s] = {a: [] for a in range(nA)}
            s_latter = desc[r][c]
            # is_done = lambda letter: letter in b'GX'

            # get_char = lambda p_d1: desc[p_d1 // MAP_COLS][p_d1 % MAP_COLS]

            if is_done(s):
                reward = REWARD_MAP[s_latter]
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            else:
                ns_up = s if r == 0 else s - MAP_COLS
                ns_right = s if c == (MAP_COLS - 1) else s + 1
                ns_down = s if r == (MAP_ROWS - 1) else s + MAP_COLS
                ns_left = s if c == 0 else s - 1

                P[s][UP] = [(1.0, ns_up, get_reward(s - MAP_COLS), is_done(s - MAP_COLS))]
                P[s][RIGHT] = [(1.0, ns_right, get_reward(s + 1), is_done(s + 1))]
                P[s][DOWN] = [(1.0, ns_down, get_reward(s + MAP_COLS), is_done(s + MAP_COLS))]
                P[s][LEFT] = [(1.0, ns_left, get_reward(s - 1), is_done(s - 1))]

            it.iternext()

        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd)

    def store_cursor(self):
        print("\033[s", end="")

    def recover_cursor(self):
        print("\033[u", end="")

    def animate_render(self, episode_i, sleep_seconds=None):
        self.store_cursor()
        print("Episode: {}".format(episode_i))
        self.render()
        if sleep_seconds is not None:
            sleep(sleep_seconds)
        self.recover_cursor()

    def render(self, mode='human', close=False):
        if close:  # 初始化环境Environment的时候不显示
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]

        state_grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(state_grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            # 对于当前状态用红色标注
            if self.s == s:
                desc[y][x] = utils.colorize(desc[y][x], "red", highlight=True)

            it.iternext()

        outfile.write("\n".join(' '.join(line) for line in desc) + "\n")

        if mode != 'human':
            return outfile
