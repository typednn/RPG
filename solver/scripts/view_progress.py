import sys
import matplotlib.pyplot as plt
import os
import pandas as p
from solver import MODEL_PATH

def view_progress(files, args):
    outs = {}
    for i in files:
        data = p.read_csv(os.path.join(MODEL_PATH, 'progress', i+'.csv'))
        outs[i] = data[args]

    outs = p.DataFrame(outs)
    outs.rolling(10).mean()[1:].plot()

    plt.savefig('progress.png')
    plt.show()

if __name__ == '__main__':
    model = sys.argv[1].split(',')
    args = sys.argv[2]
    view_progress(model, args)