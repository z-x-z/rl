'''
Description  : 
Author       : CagedBird
Date         : 2021-10-10 20:43:19
FilePath     : /rl/src/utils/show_chinese.py
'''

def show_chinese():
    from matplotlib import rcParams
    config = {
        "font.family": 'serif',
        "font.size": 14,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
        "axes.unicode_minus": False
    }
    rcParams.update(config)