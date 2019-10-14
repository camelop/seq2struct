import os, glob
from functools import reduce

# helper functions
def parse_checkpoint_path(path):
    split = os.path.split
    ckp_dir, ckp_name = split(path)
    exp_dir, exp_name = split(ckp_dir)
    step = int(ckp_name.split("-")[-1])
    return exp_name, step, path

def parse_infer_path(path):
    split = os.path.split
    ifr_dir, ifr_name = split(path)
    exp_dir, exp_name = split(ifr_dir)
    ft_split = ifr_name.split("ft")
    ft_num = int(ft_split[-1]) if len(ft_split) > 1 else 0
    step = int(ifr_name.split("step")[1][:8])
    return exp_name, step, ft_num, path

parse_evl_path = parse_infer_path

# scan to match all relevant files
def scan(base_dir="logs"):
    join = os.path.join
    ckp_pattern = reduce(join, [base_dir, "**", "model_checkpoint-*"])
    ckps = [parse_checkpoint_path(path) for path in glob.glob(ckp_pattern)]
    ifr_pattern = reduce(join, [base_dir, "**", "*infer-val-step*"])
    ifrs = [parse_infer_path(path) for path in glob.glob(ifr_pattern)]
    evl_pattern = reduce(join, [base_dir, "**", "*eval-val-step*"])
    evls = [parse_evl_path(path) for path in glob.glob(evl_pattern)]
    return ckps, ifrs, evls

if __name__ == "__main__":
    from pprint import pprint
    pprint(scan())