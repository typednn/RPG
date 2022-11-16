import numpy as np
from .envs import WorldState
from tools import CN


def parse_dict(inp):
    if isinstance(inp, list):
        out = []
        for i in inp:
            out.append(parse_dict(i))
        return out
    elif isinstance(inp, dict):
        out = CN(new_allowed=True)
        for k, v in inp.items():
            out[k] = parse_dict(v)
        return out
    try:
        x = eval(inp)
        return x
    except Exception as e:
        pass

    return inp

def load_scene(cfg_path):
    # represent scene with yaml.. 
    dict = parse_dict(CN._load_cfg_from_file(open(cfg_path, 'r')))

    import torch
    if 'Path' in dict:
        state = torch.load(dict['Path'])
    else:
        state = WorldState.sample_shapes(**dict['Shape'])
    if "Tool" in dict:
        state = state.switch_tools(**dict["Tool"])
    if "Worm" in dict:
        state.worm_scale = dict["Worm"]["action_scale"]
    return state
