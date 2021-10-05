'''
Description  : 
Author       : CagedBird
Date         : 2021-06-13 15:38:30
FilePath     : /rl/cagedbird_rl/my_codes/MC/plot_value_function.py
'''
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def plot_value_function(V, title):
    """Plots the value function as a surface plot.
        input: V (defaultdict[(palyer, dealer, ace)] = value), title
    """
    "1.确定绘图空间的大小（x，y坐标的最大最小值）"
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    "2.得到网格点坐标"
    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))
    
    def plot_surface(X, Y, Z, title=None):
        fig = plt.figure(figsize=(20, 10), facecolor='white')

        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player sum')
        ax.set_ylabel('Dealer showing')
        ax.set_zlabel('Value')
        if title: 
            ax.set_title(title)
        ax.view_init(ax.elev, -120)
        ax.set_facecolor("white")
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{}\n(No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{}\n(Usable Ace)".format(title))