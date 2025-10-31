#!/usr/bin/env python3
"""
快速测试脚本 - 根据参数构造与 run_main 一致的 setting，定位 checkpoint 并运行评测
"""
import subprocess
import sys
import os
import argparse


def build_setting(args: argparse.Namespace) -> str:
    return '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des,
        args.exp_index,
    )

def main():
    parser = argparse.ArgumentParser(description='Quick Test Runner')

    # 与 run_main 的关键字段保持一致（提供默认值）
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--model_id', type=str, default='CityA-512-48-336')
    parser.add_argument('--model_comment', type=str, default='TimeLLM-CityA')
    parser.add_argument('--model', type=str, default='TimeLLM')

    parser.add_argument('--data', type=str, default='CityA')
    parser.add_argument('--root_path', type=str, default='./dataset/city/')
    parser.add_argument('--data_path', type=str, default='CityA.csv')
    parser.add_argument('--features', type=str, default='S')

    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=336)

    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--embed', type=str, default='timeF')

    parser.add_argument('--llm_layers', type=int, default=32)
    parser.add_argument('--llm_dim', type=int, default=2048)

    parser.add_argument('--des', type=str, default='Exp')
    parser.add_argument('--exp_index', type=int, default=0)

    parser.add_argument('--checkpoints', type=str, default='./checkpoints')

    # 可选：评测脚本控制
    parser.add_argument('--summary_sample_index', type=int, default=0)
    parser.add_argument('--no_save_predictions', action='store_true')

    args = parser.parse_args()

    setting = build_setting(args)
    ckpt_dir = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
    ckpt_path = os.path.join(ckpt_dir, 'checkpoint')

    if not os.path.isfile(ckpt_path):
        print(f"未找到 checkpoint 文件: {ckpt_path}")
        print("请确认参数与训练时一致，或修改 quick_test.py 的参数后重试。")
        sys.exit(1)

    cmd = [
        'python', 'test_model_evaluation.py',
        '--checkpoint_path', ckpt_path,
        '--data', args.data,
        '--root_path', args.root_path,
        '--data_path', args.data_path,
        '--features', args.features,
        '--seq_len', str(args.seq_len),
        '--label_len', str(args.label_len),
        '--pred_len', str(args.pred_len),
        '--d_model', str(args.d_model),
        '--n_heads', str(args.n_heads),
        '--e_layers', str(args.e_layers),
        '--d_layers', str(args.d_layers),
        '--d_ff', str(args.d_ff),
        '--embed', args.embed,
        '--llm_layers', str(args.llm_layers),
        '--llm_dim', str(args.llm_dim),
        '--save_channel_summary',
        '--summary_sample_index', str(args.summary_sample_index),
    ]

    if not args.no_save_predictions:
        cmd.append('--save_predictions')

    print('运行命令:', ' '.join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
