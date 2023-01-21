import collections
import numpy as np
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas

font = {'size': 15}
import matplotlib
import os
matplotlib.rc('font', **font)

def merge_curves(x_list, y_list, bin_width=1000, max_steps=None):
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)

    idx = x.argsort()
    x = x[idx]
    y = y[idx]

    if max_steps is not None:
        idx = x <= max_steps
        x = x[idx]
        y = y[idx]
    assert (x >= 0).all()
    nbins = int(x.max() // bin_width + 1)
        
    n, _ = np.histogram(x, bins=nbins)
    sy, _ = np.histogram(x, bins=nbins, weights=y)
    sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
    xx, _ = np.histogram(x, bins=nbins, weights=x)

    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    xx = xx / n
    idx = xx>0
    return xx[idx], mean[idx], std[idx]


def smooth(y, smoothingWeight=0.95):
    y_smooth = []
    last = y[0]
    for i in range(len(y)):
        y_smooth.append(last * smoothingWeight + (1 - smoothingWeight) * y[i])
        last = y_smooth[-1]
    return np.array(y_smooth)


def plot_curve_with_shade(ax, x, mean, std, label, color='green', smoothingWeight=0):
    
    #y_smooth = mean
    y_smooth = smooth(mean, smoothingWeight)
    std_smooth = smooth(std, smoothingWeight) * 0.3
    ax.fill_between(x, (y_smooth - std_smooth).clip(0, np.inf), y_smooth + std_smooth, facecolor=color, alpha=0.2)
    ax.plot(x, y_smooth, color=color, label=label, linewidth=3.)


KEYS = defaultdict(lambda: 'success')
MAX_STEPS = defaultdict(lambda: 2000000)


def get_df_xy(df, x, y):
    x = df[x]
    y = df[y]
    data = np.stack([x, y]).T
    data = data[~np.isnan(y)]
    return data[:, 0], data[:, 1]

def read_baseline_result(env_name, method):
    env_name = dict(
        hammer='AdroitHammer',
        ant='AntPush',
        block='BlockPush',
        cabinet='Cabinet',
        stickpull='MWStickPull',
    )[env_name]

    x_keys = dict(sac='env_n_samples', tdmpc='env_steps')

    if method != 'sac':
        query = f"data/collected_results/{method}/{env_name}/*/progress.csv"
    else:
        query = f"data/collected_results/{method}/{env_name}/*/*/progress.csv"
    xs = []
    ys = []
    for path in list(glob.iglob(query)):
        try:
            df = pandas.read_csv(path)
        except Exception as e:
            # print(e)
            continue

        x, y = get_df_xy(df, x_keys[method], 'eval_success_rate')
        # print(path, len(x))
        xs.append(x); ys.append(y)
    return xs, ys

def read_wandb_result(env_name, method):
    xs, ys = [], []
    for path in list(glob.iglob(f"data/wandb/{env_name}/{method}/*.csv")):
        try:
            df = pandas.read_csv(path)
        except Exception as e:
            # print(e)
            continue

        x, y = get_df_xy(df, 'total_steps', 'success')
        # print(path, len(x))
        xs.append(x); ys.append(y)
    return xs, ys


def plot_env(ax: plt.Axes, env_name):
    results = dict(
        sac=read_baseline_result(env_name, 'sac'),
        tdmpc=read_baseline_result(env_name, 'tdmpc'),
        mbsac=read_wandb_result(env_name, 'mbsac'),
        rpg=read_wandb_result(env_name, 'rpg'),
    )
    
    #print(len(results['sac'][0]))
    idx = 0
    for k, v in results.items():
        if len(v[0]) > 0:
            print(k, len(v[0]), [len(x) for x in v[0]])
            plot_curve_with_shade(ax, *merge_curves(v[0], v[1], 10000, MAX_STEPS[env_name]),
                                label=k + f" ({len(v[0])} runs)", smoothingWeight=0.05, color = 'C' + str(idx))
        idx += 1
        
    
    ax.legend(loc=2)
    ax.set_title(env_name)
    ax.set_xlabel("interactions (M)"); 
    ax.set_ylabel("success rate")
    ax.grid()
    

    
if __name__ == '__main__':
    #plt.figure(figsize=(6, 6))
    envs = ['stickpull', 'cabinet', 'block']
    fig, axs = plt.subplots(1, len(envs), figsize=(6 * len(envs), 6))
    for ax, env_name in zip(axs, envs):
        plot_env(ax, env_name)
    plt.savefig('test.png', dpi=300)