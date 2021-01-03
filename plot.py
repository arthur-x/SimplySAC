import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_curve(env):
    min_l = 2001
    ret_list = []
    for s in range(5):
        df = pd.read_csv('saves/' + str(env+1) + '/log' + str(s+1) + '.csv')
        ret = df[['return']].to_numpy().transpose(1, 0)[0]
        if len(ret) < min_l:
            min_l = len(ret)
        for i in range(len(ret) - 1):
            ret[i + 1] = ret[i] * 0.9 + ret[i + 1] * 0.1
        ret_list.append(ret)
    data = np.zeros((5, min_l))
    for s in range(5):
        data[s, :] = ret_list[s][:min_l]
    mean = np.mean(data, axis=0)
    mini = np.min(data, axis=0)
    maxi = np.max(data, axis=0)
    stamps = np.array([i * 1e-3 for i in range(min_l)])
    plt.plot(stamps, mean, label='SAC', lw=1.0)
    plt.fill_between(stamps, mini, maxi, alpha=0.2)
    plt.title(env_list[env])
    plt.xlabel('number of environment steps (x $\mathregular{10^6}$)')
    plt.ylabel('return')
    plt.xlim(0, 2)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    env_list = ['Walker2d-v2', 'HalfCheetah-v2', 'Ant-v2', 'Humanoid-v2', 'AntBulletEnv-v0', 'HumanoidBulletEnv-v0']
    mpl.style.use('seaborn')
    for env in range(6):
        plot_curve(env)
