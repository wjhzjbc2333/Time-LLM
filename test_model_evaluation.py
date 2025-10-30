#!/usr/bin/env python3
"""
TimeLLM 模型测试评估脚本
用于加载训练好的模型并进行测试评估
"""
import argparse
import torch
import numpy as np
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from accelerate import Accelerator

# 设置环境变量
os.environ["USE_LIBUV"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from models import TimeLLM
from data_provider.data_factory import data_provider
from utils.tools import load_content

# 定义评估指标函数
def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true)) * 100

def MSPE(pred, true):
    return np.mean(((pred - true) / true) ** 2) * 100

def create_args():
    """创建测试参数，需要与训练时的参数保持一致"""
    parser = argparse.ArgumentParser(description='TimeLLM Model Evaluation')
    
    # 基本配置 - 需要与训练时保持一致
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--model_id', type=str, default='ETTh1_512_96', help='model id')
    parser.add_argument('--model_comment', type=str, default='TimeLLM-ETTh1', help='model comment')
    parser.add_argument('--model', type=str, default='TimeLLM')
    parser.add_argument('--seed', type=int, default=2021)
    
    # 数据加载配置
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--loader', type=str, default='modal', help='dataset type')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    
    # 预测任务配置
    parser.add_argument('--seq_len', type=int, default=512, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--percent', type=int, default=100, help='dataset percent for training split')
    
    # 模型定义配置
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=0)
    parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model')
    parser.add_argument('--llm_dim', type=int, default=2048, help='LLM model dimension')
    parser.add_argument('--llm_layers', type=int, default=32)
    
    # 测试配置
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision', default=False)
    
    # 模型路径配置
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                       help='path to the trained model checkpoint')
    parser.add_argument('--save_predictions', action='store_true', 
                       help='save predictions to file')
    parser.add_argument('--plot_results', action='store_true', 
                       help='plot prediction results')
    
    return parser.parse_args()

def load_trained_model(args, checkpoint_path):
    """加载训练好的模型"""
    print(f"正在加载模型: {checkpoint_path}")
    
    # 创建模型
    model = TimeLLM.Model(args).to(torch.bfloat16)
    
    # 检查checkpoint文件是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
    
    # 加载权重
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print("✅ 模型权重加载成功")
    except Exception as e:
        print(f"❌ 模型权重加载失败: {e}")
        raise
    
    return model

def evaluate_model(args, model, test_loader, accelerator):
    """评估模型性能"""
    print("开始模型评估...")
    
    model.eval()
    total_loss = []
    total_mae_loss = []
    all_predictions = []
    all_targets = []
    
    criterion = torch.nn.MSELoss()
    mae_metric = torch.nn.L1Loss()
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader, desc="Evaluating")):
            # 数据预处理
            batch_x = batch_x.float().to(accelerator.device).to(torch.bfloat16)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(accelerator.device).to(torch.bfloat16)
            batch_y_mark = batch_y_mark.float().to(accelerator.device).to(torch.bfloat16)
            
            # 解码器输入
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device).to(torch.bfloat16)
            
            # 模型预测
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # 处理输出
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            
            # 计算损失
            loss = criterion(outputs, batch_y.to(accelerator.device))
            mae_loss = mae_metric(outputs, batch_y.to(accelerator.device))
            
            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())
            
            # 收集预测结果
            all_predictions.append(outputs.cpu().float().numpy())
            all_targets.append(batch_y.numpy())
    
    # 计算平均损失
    avg_loss = np.mean(total_loss)
    avg_mae_loss = np.mean(total_mae_loss)
    
    # 合并所有预测结果
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return avg_loss, avg_mae_loss, all_predictions, all_targets

def calculate_metrics(predictions, targets):
    """计算各种评估指标"""
    print("计算评估指标...")
    
    # 确保数据形状一致
    if predictions.shape != targets.shape:
        print(f"警告: 预测形状 {predictions.shape} 与目标形状 {targets.shape} 不匹配")
        min_len = min(predictions.shape[1], targets.shape[1])
        predictions = predictions[:, :min_len, :]
        targets = targets[:, :min_len, :]
    
    # 计算各种指标
    mse = MSE(predictions, targets)
    rmse = RMSE(predictions, targets)
    mae = MAE(predictions, targets)
    mape = MAPE(predictions, targets)
    mspe = MSPE(predictions, targets)
    rse = RSE(predictions, targets)
    corr = CORR(predictions, targets)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'MSPE': mspe,
        'RSE': rse,
        'CORR': corr
    }

def plot_results(predictions, targets, save_path='./results'):
    """绘制预测结果"""
    print("绘制预测结果...")
    
    os.makedirs(save_path, exist_ok=True)
    
    # 选择前几个样本进行可视化
    num_samples = min(5, predictions.shape[0])
    num_features = predictions.shape[2]
    
    fig, axes = plt.subplots(num_samples, num_features, figsize=(15, 3*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if num_features == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_samples):
        for j in range(num_features):
            ax = axes[i, j]
            
            # 绘制真实值和预测值
            ax.plot(targets[i, :, j], label='True', color='blue', alpha=0.7)
            ax.plot(predictions[i, :, j], label='Predicted', color='red', alpha=0.7)
            
            ax.set_title(f'Sample {i+1}, Feature {j+1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'prediction_results.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"预测结果图已保存到: {os.path.join(save_path, 'prediction_results.png')}")

def save_predictions(predictions, targets, save_path='./results'):
    """保存预测结果"""
    print("保存预测结果...")
    
    os.makedirs(save_path, exist_ok=True)
    
    # 保存为numpy文件
    np.save(os.path.join(save_path, 'predictions.npy'), predictions)
    np.save(os.path.join(save_path, 'targets.npy'), targets)
    
    # 保存为CSV文件（仅第一个特征）
    if predictions.shape[2] > 0:
        pred_df = pd.DataFrame(predictions[:, :, 0])
        target_df = pd.DataFrame(targets[:, :, 0])
        
        pred_df.to_csv(os.path.join(save_path, 'predictions.csv'), index=False)
        target_df.to_csv(os.path.join(save_path, 'targets.csv'), index=False)
    
    print(f"预测结果已保存到: {save_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("TimeLLM 模型测试评估")
    print("=" * 60)
    
    # 解析参数
    args = create_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 初始化accelerator
    accelerator = Accelerator()
    
    # 加载数据
    print("加载测试数据...")
    test_data, test_loader = data_provider(args, 'test')
    print(f"测试数据加载完成，共 {len(test_loader)} 个批次")
    
    # 加载模型
    model = load_trained_model(args, args.checkpoint_path)
    
    # 准备模型
    model = accelerator.prepare(model)
    
    # 评估模型
    print("\n开始模型评估...")
    start_time = time.time()
    
    avg_loss, avg_mae_loss, predictions, targets = evaluate_model(args, model, test_loader, accelerator)
    
    # 计算评估指标
    metrics = calculate_metrics(predictions, targets)
    
    end_time = time.time()
    evaluation_time = end_time - start_time
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"评估时间: {evaluation_time:.2f} 秒")
    print(f"平均MSE损失: {avg_loss:.6f}")
    print(f"平均MAE损失: {avg_mae_loss:.6f}")
    print("\n详细指标:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, np.ndarray):
            if metric_value.size == 1:
                metric_value = metric_value.item()
            else:
                metric_value = metric_value.mean()
        print(f"  {metric_name}: {metric_value:.6f}")
    
    # 保存结果
    if args.save_predictions:
        save_predictions(predictions, targets)
    
    # 绘制结果
    if args.plot_results:
        plot_results(predictions, targets)
    
    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()
