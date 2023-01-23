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
    door=dict(
        rpg=['guwbs8n3', 'xaqt9na9', '2hqi924c', '1ap4457z'],
        mbsac=['9zrks8f2', 'row6vwfk', 'sglqybdb', '5rcqeg67'],
    ),
    basket=dict(
        rpg=['br432idf', '5d9gqf69', 'v8cgw2k5', 'evdbrymj'],
        mbsac=['3b1ah01n', 'ni4z9bqx', 'oa3eeu5v', '1o6la58u'],
    ),


    gapexplore=dict(
        discrete=['sr28zf2d', '9dsw1kd4', '8r2izj37', 'wdu1kiqr', 'niflfzpl', 'dj0yl14z'],
        gaussian=['sgt89o05', '63xovceh', '4rkgypwi', 'g7h113pn', 'dstuhayw', 'datstv8g'],
        mbsac=['wst9rdma', 'juxzki6j', '55zrzlf6', 'c0ki4phi', 'ns1lde2p', 'bl5qygt1'],
        gaussian05=['f36qxioj', '7fr4ydrh', 'wwofyx0z'],
        gaussian0001=['kfgqeg7c', 'ag2n17dc', 'e832ufwl', 'q2w1bezy', 'mbvvawgm', 'elou48ql'],
        gaussian0=['xaq8cwrf', 'ee83kcpl', '644bu1ut', 'cxkkpv0m', 'cq86vioo', 'wah4bo00'],
        gaussian005=['fz18vlk8', '3352pxiq', '1lbtmgnx', 'j3v2lxnf', 'egaptasv', '6dwu95nv'],
        gaussiand3=['6ndjhh76', 'xeqgwaou', '2gxn73fl', 'of2g238e', 'jkzmv65w', 'ctcdg1nn'],
        gaussiand6=['j9xn3823', '4eswf4b0', 'tup4123d', 'h2ag4xbp', 'gdlkxca5', 'pwpig51t'],
        gaussiand1=['nfm4zi9g', '93i64mv6', '3uukvt6u', 'wjee13tv', 'h130ktba', 'fprhq5j1'],
        mpc=['uk7bxti8', 'u6xtv0eg', '3vx0zkjo', 'cwdwphbi', 'e6oof9o6', '5msztf2o'],
        flow=['rc8r3nrs', 'otlrp3qt', '4lz18l3c', 'sbb7o6h5', 'f7lyevpd', 'f0ptehia'],
        gmm=['zyxnuaks', 'u08gevjd', '5sc36fsk', 'nzeh1eed', 'i09ldafx', 'eineqnz1'],
        rndd0=['ttrid9hp', 's3yq7dcc', '5t8k34ur', 'xyk57f8l', 'jm8p1o9t', '5qwg4g48'],
        rndnobuf=['uqzkebsa', 'vt6twug9', 'kbu1kiwu', 'tw4f66lp', 'c3opl4in', 'n029tbse'],
        #rndnonorm=[],
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