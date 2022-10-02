import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as p
from solver import MODEL_PATH
import argparse
import re



def readall(filename):
    assert os.path.isdir(filename), f"{filename} is not a directory"
    outs = {}
    for root, dirs, files in os.walk(filename):
        #print(root, dirs,files)
        if 'progress.csv' in files:
            try:
                curve = pd.read_csv(os.path.join(root, 'progress.csv'))
            except:
                continue
            relpath = os.path.relpath(root, filename)
            outs[relpath] = curve
            print(len(curve))
    return outs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exps")
    parser.add_argument("--key", default='coverage')
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--n", default=None, type=str)
    parser.add_argument("--filter", default=None, type=str)
    parser.add_argument("--clamp", default=None, type=str)
    args = parser.parse_args()

    mean = {}
    std = {}
    if args.exps == 'all':
        assert args.prefix is not None
        args.exps = os.listdir(os.path.join(MODEL_PATH, args.prefix))
    else:
        args.exps = args.exps.split(',')
    if args.filter is not None:
        prog = re.compile(args.filter)
        args.exps = [i for i in args.exps if prog.match(i)]
    for root in args.exps:
        if args.prefix is not None:
            root = os.path.join(args.prefix, root)

        prefix = root
        curves = readall(os.path.join(MODEL_PATH, root))
        data = {}
        for i in curves:
            data[i] = curves[i][args.key]
        if len(data) == 0:
            continue
        data = pd.DataFrame(data)
        if args.n is not None:
            data = data[eval(f"slice({args.n})")]
        if args.clamp is not None:
            data = data.clip(*eval(args.clamp))
        mean = data.mean(axis=1).to_numpy()
        std = data.std(axis=1).to_numpy()
        print(mean.shape, std.shape, data.index.shape)
        if np.isnan(std).any():
            std = None
        #plt.errorbar(data.index, mean, std, label=prefix)
        line = plt.plot(data.index, mean, label=prefix)
        if std is not None:
            plt.fill_between(data.index, mean-std, mean+std, alpha=0.3, color=line[0].get_color())
    #plt.errorbar(x, y, e, linestyle='None', marker='^')
    #outs = pd.DataFrame(outs).mean(axis=1)
    #outs.plot()
    plt.legend()
    #plt.show()
    plt.savefig('tmp.png')


if __name__ == '__main__':
    main()