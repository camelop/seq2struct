import os, glob
import sys
from tracker import track




def get_checkpoint_config_path(ckp_path):
    ckp_dir, ckp_name = os.path.split(ckp_path)
    config_path = glob.glob(os.path.join(ckp_dir, "*config-*.json"))
    if len(config_path) < 1:
        raise FileNotFoundError("Config file not found for checkpoint: {}".format(ckp_path))
    if len(config_path) > 1:
        raise FileExistsError("Multiple config file for checkpoint: {}".format(ckp_path))
    return config_path[0]

def generate_infer_command(base_dir="logs", use_force_no_ft=True):
    infer_command_template = ('python infer.py --config {config_path} '
                '--logdir logs/{exp_name} '
                '--output logs/{exp_name}/infer-val-step{step:08d}-bs1.jsonl '
                '--step {step} --section val --beam-size 1 {force_no_ft}')
    force_no_ft = "--force-no-ft" if use_force_no_ft else ""
    ckp_path, ckp_evl = track(base_dir)
    commands = []
    for ckp in sorted(ckp_path.keys()):
        status = sum(_ is not None for _ in ckp_evl[ckp][0]) # 0: Nothing, 1: inferred, 2: evaluated
        if status > 0:
            continue
        exp_name, step = ckp
        path = ckp_path[ckp]
        config_path = get_checkpoint_config_path(path)
        infer_command = infer_command_template.format(
            config_path=config_path,
            exp_name=exp_name,
            step=step,
            force_no_ft=force_no_ft
        )
        commands.append(infer_command)
    return commands

def generate_eval_command(base_dir="logs"):
    eval_command_template =  ('python eval.py --config {config_path} '
                '--logdir logs/{exp_name} '
                '--inferred {infer_path} '
                '--output logs/{exp_name}/{ft_flag}eval-val-step{step:08d}-bs1.jsonl{ft_end} '
                '--section val')
    ckp_path, ckp_evl = track(base_dir)
    commands = []
    for ckp in sorted(ckp_path.keys()):
        exp_name, step = ckp
        path = ckp_path[ckp]
        config_path = get_checkpoint_config_path(path)
        for ft_num, (infer_path, eval_path) in enumerate(ckp_evl[ckp]):
            if eval_path is not None:
                continue
            if infer_path is None:
                break
            ft_flag, ft_end = "", ""
            if ft_num > 1 or "ft" in os.path.split(infer_path)[1]:
                ft_flag, ft_end = "ft-", ".ft{}".format(ft_num)
            eval_command = eval_command_template.format(
                config_path=config_path,
                exp_name=exp_name,
                infer_path=infer_path,
                step=step,
                ft_flag=ft_flag,
                ft_end=ft_end
            )
            commands.append(eval_command)
    return commands

if __name__ == "__main__":
    cmd_type = sys.argv[1]
    if cmd_type == "infer":  # infer without finetune
        print("export CUDA_VISIBLE_DEVICES=-1")
        for cmd in generate_infer_command(use_force_no_ft=True):
            print(cmd)
    elif cmd_type == "eval":
        print("export CUDA_VISIBLE_DEVICES=-1")
        for cmd in generate_eval_command():
            print(cmd)
    else:
        raise NotImplementedError