import wandb
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
api = wandb.Api()

results = dict(
    stickpull=dict(
        rpg=['sswo3gu9', 'qbgz5rgw', 'hhyvwxgg', 'csht63sz', 'zzn586pj'],
        mbsac=['2i7c0xvi', 'bl1i7dq4', 'xh9ahff4', '07zn0sft'],
    ),
    cabinet=dict(
        rpg=['mam86fgv', 'a7tae934', 'uzji2ksc', '1b9kgqbs', 'v6ryfx53', 'h1wbr6ax'],
        mbsac=['goj926av', 'f4sx5rsd', '9rsqbq67', 'q858ft9t', 'oyr4qmtb']
    ),
    block=dict(
        rpg=['lgz9k406', 'd7o37qic', 'yen88bee', 'qfwjbjxc', '3r6drtye', '7chiegoc'],
        mbsac=['33mp99t7', 'd2082yfk', 'hjkrcw4q', '0bahdeev', 'rks1e3iw', 'rjsld8bo'],
    ),
    hammer=dict(
        rpg=['lxt5ouxl', '8mg72vdx', 'vpela3n1', 'hbnqrdfn', '8g2en9hc', '5dnz5hkp'],
        mbsac=['mlnuyk4x', '6cbeqo8q', '7t5zwzo7', '2ysu2ah7'],
    ),
    densecabinet=dict(
        rpg=['l74istv9', 'a1mwbhc2', 'mxk0e15e', 'nezxnnsm'],
        mbsac=['czbmv6kz', '80fo7yfs', 'rdbxkafw', 's098uexc', 'zmr3ajup', 'j1e5wnf6'],
    ),
    denseantpush=dict(
        mbsac=['qrk7pms9', 'imuhdlx3', 'rologrt9', 'y6f90qiv', '2vaesy7r', '0i8xrgw0'],
        rpg=['c7t6092a', '80bll73c', 'igkxhusp', 'pofu2kez'],
    ),
    denseantfall=dict(
        mbsac=['ymbql7z2', '0avkweg1', 'yr7fqyns', 'nmdap4k3'],
        rpg= [],
    ),
    kitchen=dict(
        rpg=['kuoq1sgj', 'esn4a9dc', 'jk5klnmk', '7z71xou5'],
        mbsac=['wgwnxg18', 'cjl9c2jh', 'sugg7aly', 'uqlwi5jl'],
    ),
    antpush=dict(
        mbsac=['208y8mza', '6dw1lmn3'],
        #rpg=['vwppvqay', 'obbvdzv7']
        rpg = ['d9w7ftjd', '7k1g89ok']
    ),
    antfall=dict(
        rpg=['tp8vynv0', 'szm0fdhi'],
        mbsac=['dg8eqb43', 'rdux2as3'],
    ),
)

def download_run(rid, path):
    os.makedirs(path, exist_ok=True)
    run = api.run(f"/feyat/maze_exp/{rid}")
    history = run.scan_history()
    ds = []
    for i in history:
        ds.append(dict(i))
    print(len(ds))
    pd.DataFrame(ds).to_csv(os.path.join(path, f'{rid}.csv'))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, default='antfall')
    args = parser.parse_args()


    save_path = 'data/wandb/'
    name = args.name

    for method, rids in results[name].items():
        for rid in rids:
            download_run(rid, os.path.join(save_path, name, method))

if __name__ == '__main__':
    main()