import numpy as np


def generate_map(map_size=(4, 4), holes=3, goals=1, seed=None):
    """ A generator that generate maps for GridworldEnv
    """
    MAP_ROWS = map_size[0]
    MAP_COLS = map_size[1]
    map_points = MAP_ROWS * MAP_COLS
    if (holes + goals + 1 > map_points):
        print("Error: Holes({0}) add goals({1}) larger than {2}({3} * {4} -1)!"\
            .format(holes, goals, map_points - 1, MAP_ROWS, MAP_COLS))
        raise EnvironmentError
    if seed:
        np.random.seed(seed)
    shuffle_d1cors = np.arange(map_points)
    np.random.shuffle(shuffle_d1cors)
    "一维坐标转化为二维坐标"
    d1_to_d2 = lambda d1_cor: (d1_cor // MAP_COLS, d1_cor % MAP_COLS)
    start_d2cor = d1_to_d2(shuffle_d1cors[0])
    holes_d2cors = [d1_to_d2(d1cor) for d1cor in shuffle_d1cors[1:1 + holes]]
    goals_d2cors = [d1_to_d2(d1cor) for d1cor in shuffle_d1cors[1 + holes:1 + holes + goals]]

    "绘制地图"
    # 注意不要使用*法创建相应的数组，因为可能出现引用错误
    res_map = [['O' for j in range(MAP_COLS)] for i in range(MAP_ROWS)]
    res_map[start_d2cor[0]][start_d2cor[1]] = 'S'  # 设置开始位置
    for r, c in holes_d2cors:  # 设置陷阱位置
        res_map[r][c] = 'X'
    for r, c in goals_d2cors:  # 设置成果位置
        res_map[r][c] = 'G'
    for i in range(len(res_map)):
        res_map[i] = "".join(res_map[i])

    return res_map


def generate_map_by_content(content):
    '''通过实际地图获得数据地图
        return map_size, map
    '''
    content = """O X O O X O
O G O X X O
O X X X O O
O O O O O O
O X O O X O
O O O O X S"""
    res_map = []
    for line in content.split("\n"):
        res_map.append(line.replace(" ", ""))
    return (len(res_map), len(res_map[0])), res_map

def test_map_generator():
    np.random.seed()
    map66 = generate_map((10, 10), 30, 1)
    print(map66)
    for line in map66:
        print(line)
