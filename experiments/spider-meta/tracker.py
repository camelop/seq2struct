from scanner import scan

def track(base_dir="logs", max_ft=None):
    ckps, ifrs, evls = scan(base_dir)
    ckp_path = {}
    ckp_evl = {}
    if not max_ft:
        max_ft = max(ft_num for _, _, ft_num, _ in ifrs) + 1
    for exp_name, step, path in ckps:
        ckp = (exp_name, step)
        ckp_path[ckp] = path
        ckp_evl[ckp] = [(None, None)] * max_ft
    for exp_name, step, ft_num, path in ifrs:
        ckp = (exp_name, step)
        ckp_evl[ckp][ft_num] = (path, None)
    for exp_name, step, ft_num, path in evls:
        ckp = (exp_name, step)
        ckp_evl[ckp][ft_num] = (ckp_evl[ckp][ft_num][0], path)
    return ckp_path, ckp_evl

def display_track_result(ckp_path, ckp_evl):
    print("All checkpoints:")
    for k in sorted(ckp_path.keys()):
        v = ckp_path[k]
        line = "{:30}[{:6}]: {:40}".format(k[0], k[1], v if len(v)<=37 else "..."+v[-37:])
        for ft_num, sub in enumerate(ckp_evl[k]):
            if sub is None or sub[0] is None:
                mark = " ."
            elif sub[1] is None:
                mark = " ?"
            else:
                mark = " *"
            line = line + mark
        print(line)

if __name__ == "__main__":
    display_track_result(*track())