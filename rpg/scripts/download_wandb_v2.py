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
    )
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
    save_path = 'data/wandb/'
    name = 'block'

    for method, rids in results[name].items():
        for rid in rids:
            download_run(rid, os.path.join(save_path, name, method))

if __name__ == '__main__':
    main()