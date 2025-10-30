#!/usr/bin/env python3
"""
快速测试脚本 - 使用找到的checkpoint文件
"""
import subprocess
import sys
import os
import glob


def find_first_checkpoint():
    ckpt_dirs = glob.glob(os.path.join(".", "checkpoints", "*"))
    for d in ckpt_dirs:
        ckpt = os.path.join(d, "checkpoint")
        if os.path.isfile(ckpt):
            return ckpt
    return None


def parse_args_from_setting_dir(checkpoint_path):
    # directory name contains setting string before last '-' (model_comment suffix)
    setting_dir = os.path.basename(os.path.dirname(checkpoint_path))
    setting_core = setting_dir.rsplit('-', 1)[0]
    parts = setting_core.split('_')
    
    # expected indices from run_main setting format
    # 0:long 1:term 2:forecast 3:model_id 4:model 5:data then keyed tokens
    parsed = {
        'data': parts[5] if len(parts) > 5 else 'ETTh1',  # 默认ETTh1
        'features': 'M',
        'seq_len': 512,
        'label_len': 48,
        'pred_len': 96,
        'd_model': 32,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'd_ff': 128,
        'factor': 3,
        'embed': 'timeF',
        'llm_layers': 32,  # 默认LLM层数
        'llm_dim': 2048,  # 默认LLM维度
    }
    for p in parts:
        if p.startswith('ft'):
            parsed['features'] = p[2:]
        elif p.startswith('sl'):
            parsed['seq_len'] = int(p[2:])
        elif p.startswith('ll'):
            parsed['label_len'] = int(p[2:])
        elif p.startswith('pl'):
            parsed['pred_len'] = int(p[2:])
        elif p.startswith('dm'):
            parsed['d_model'] = int(p[2:])
        elif p.startswith('nh'):
            parsed['n_heads'] = int(p[2:])
        elif p.startswith('el'):
            parsed['e_layers'] = int(p[2:])
        elif p.startswith('dl'):
            parsed['d_layers'] = int(p[2:])
        elif p.startswith('df'):
            parsed['d_ff'] = int(p[2:])
        elif p.startswith('fc'):
            parsed['factor'] = int(p[2:])
        elif p.startswith('eb'):
            parsed['embed'] = p[2:]
        # 注意：llm_layers 和 llm_dim 通常不在目录名中，需要从训练参数推断
        # 这里我们使用默认值，用户可以通过命令行参数覆盖
    return parsed

def main():
    ckpt = find_first_checkpoint()
    if ckpt is None:
        print("未找到checkpoint文件，请先训练模型。")
        sys.exit(1)

    parsed = parse_args_from_setting_dir(ckpt)

    data_path = f"{parsed['data']}.csv"
    root_path = "./dataset/ETT-small"

    cmd = [
        "python", "test_model_evaluation.py",
        "--checkpoint_path", ckpt,
        "--data", parsed['data'],
        "--root_path", root_path,
        "--data_path", data_path,
        "--features", parsed['features'],
        "--seq_len", str(parsed['seq_len']),
        "--label_len", str(parsed['label_len']),
        "--pred_len", str(parsed['pred_len']),
        "--d_model", str(parsed['d_model']),
        "--n_heads", str(parsed['n_heads']),
        "--e_layers", str(parsed['e_layers']),
        "--d_layers", str(parsed['d_layers']),
        "--d_ff", str(parsed['d_ff']),
        "--embed", parsed['embed'],
        "--llm_layers", str(parsed['llm_layers']),
        "--llm_dim", str(parsed['llm_dim']),
        "--save_predictions",
        "--plot_results",
    ]
    
    print("运行命令:", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
