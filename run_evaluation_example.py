#!/usr/bin/env python3
"""
TimeLLM 模型评估使用示例
"""
import os
import glob
import subprocess
import sys

def find_checkpoint_files():
    """查找checkpoint文件"""
    checkpoint_dirs = glob.glob("./checkpoints/*/")
    checkpoint_files = []
    
    for dir_path in checkpoint_dirs:
        checkpoint_path = os.path.join(dir_path, "checkpoint")
        if os.path.exists(checkpoint_path):
            checkpoint_files.append(checkpoint_path)
    
    return checkpoint_files

def show_usage_examples():
    """显示使用示例"""
    print("=" * 80)
    print("TimeLLM 模型评估使用指南")
    print("=" * 80)
    
    print("\n1. 基本使用方法:")
    print("   python test_model_evaluation.py --checkpoint_path <checkpoint文件路径>")
    
    print("\n2. 完整参数示例:")
    print("   python test_model_evaluation.py \\")
    print("       --checkpoint_path ./checkpoints/long_term_forecast_ETTh1_512_96_TimeLLM_ETTh1_ftM_sl512_ll48_pl96_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_Exp_0-TimeLLM-ETTh1/checkpoint \\")
    print("       --data ETTh1 \\")
    print("       --root_path ./dataset/ETT-small/ \\")
    print("       --data_path ETTh1.csv \\")
    print("       --seq_len 512 \\")
    print("       --pred_len 96 \\")
    print("       --d_model 32 \\")
    print("       --d_ff 128 \\")
    print("       --llm_dim 2048 \\")
    print("       --llm_layers 32 \\")
    print("       --save_predictions \\")
    print("       --plot_results")
    
    print("\n3. 查找现有checkpoint文件:")
    checkpoint_files = find_checkpoint_files()
    if checkpoint_files:
        print("   找到以下checkpoint文件:")
        for i, file_path in enumerate(checkpoint_files, 1):
            print(f"   {i}. {file_path}")
    else:
        print("   未找到checkpoint文件，请先训练模型")
    
    print("\n4. 参数说明:")
    print("   必需参数:")
    print("     --checkpoint_path: 训练好的模型权重文件路径")
    print("   ")
    print("   可选参数:")
    print("     --data: 数据集名称 (默认: ETTh1)")
    print("     --root_path: 数据文件根路径 (默认: ./dataset/ETT-small/)")
    print("     --data_path: 数据文件名 (默认: ETTh1.csv)")
    print("     --seq_len: 输入序列长度 (默认: 512)")
    print("     --pred_len: 预测序列长度 (默认: 96)")
    print("     --d_model: 模型维度 (默认: 32)")
    print("     --d_ff: 前馈网络维度 (默认: 128)")
    print("     --llm_dim: LLM模型维度 (默认: 2048)")
    print("     --llm_layers: LLM层数 (默认: 32)")
    print("     --save_predictions: 保存预测结果")
    print("     --plot_results: 绘制预测结果图")
    
    print("\n5. 输出文件:")
    print("   - 控制台输出: 评估指标结果")
    print("   - ./results/predictions.npy: 预测结果 (numpy格式)")
    print("   - ./results/targets.npy: 真实值 (numpy格式)")
    print("   - ./results/predictions.csv: 预测结果 (CSV格式)")
    print("   - ./results/targets.csv: 真实值 (CSV格式)")
    print("   - ./results/prediction_results.png: 预测结果可视化图")
    
    print("\n6. 评估指标说明:")
    print("   - MSE: 均方误差 (Mean Squared Error)")
    print("   - RMSE: 均方根误差 (Root Mean Squared Error)")
    print("   - MAE: 平均绝对误差 (Mean Absolute Error)")
    print("   - MAPE: 平均绝对百分比误差 (Mean Absolute Percentage Error)")
    print("   - MSPE: 均方百分比误差 (Mean Squared Percentage Error)")
    print("   - RSE: 相对平方误差 (Relative Squared Error)")
    print("   - CORR: 相关系数 (Correlation)")
    
    print("\n" + "=" * 80)

def create_quick_test_script():
    """创建快速测试脚本"""
    checkpoint_files = find_checkpoint_files()
    
    if not checkpoint_files:
        print("❌ 未找到checkpoint文件，请先训练模型")
        return
    
    # 使用第一个找到的checkpoint文件
    checkpoint_path = checkpoint_files[0]
    
    script_content = f'''#!/usr/bin/env python3
"""
快速测试脚本 - 使用找到的checkpoint文件
"""
import subprocess
import sys

def main():
    cmd = [
        "python", "test_model_evaluation.py",
        "--checkpoint_path", "{checkpoint_path}",
        "--save_predictions",
        "--plot_results"
    ]
    
    print("运行命令:", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
'''
    
    with open("quick_test.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print(f"✅ 已创建快速测试脚本: quick_test.py")
    print(f"   使用checkpoint文件: {checkpoint_path}")
    print("   运行命令: python quick_test.py")

if __name__ == "__main__":
    show_usage_examples()
    create_quick_test_script()
