#!/usr/bin/env python3
"""
修改后的test.py，支持加载预训练权重
"""
from accelerate import Accelerator
from models import TimeLLM
import torch
from torch import nn, optim
from data_provider_pretrain.data_factory import data_provider
import argparse
import time
import numpy as np
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content
import os
os.environ["USE_LIBUV"] = "0"

def load_pretrained_model(args, checkpoint_path):
    """加载预训练模型权重"""
    # 创建模型
    model = TimeLLM.Model(args).to(torch.bfloat16)
    
    # 检查checkpoint文件是否存在
    if os.path.exists(checkpoint_path):
        print(f"正在加载预训练权重: {checkpoint_path}")
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print("✅ 预训练权重加载成功")
        except Exception as e:
            print(f"❌ 预训练权重加载失败: {e}")
            print("将使用随机初始化的权重")
    else:
        print(f"⚠️  未找到预训练权重文件: {checkpoint_path}")
        print("将使用随机初始化的权重")
    
    return model

def main():
    accelerator = Accelerator()

    '''Arguments'''
    parser = argparse.ArgumentParser(description='Time-LLM')
    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='test')
    parser.add_argument('--model_comment', type=str, default='none', help='prefix when saving test results')
    parser.add_argument('--model', type=str, default='TimeLLM')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    
    # 添加预训练权重路径参数
    parser.add_argument('--pretrained_path', type=str, default='', 
                       help='预训练模型权重路径，例如: ./checkpoints/long_term_forecast_test_TimeLLM_ETTm1_ftM_sl96_ll48_pl96_dm16_nh8_el2_dl1_df32_fc1_ebtimeF_test_0-none/checkpoint')

    # data loader
    parser.add_argument('--data_pretrain', type=str, default='ETTm1', help='dataset type')
    parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--data_path_pretrain', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; '
                             'M:multivariate predict multivariate, S: univariate predict univariate, '
                             'MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--loader', type=str, default='modal', help='dataset type')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, '
                             'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                             'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=0, help='')
    parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
    parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--percent', type=int, default=100)
    args = parser.parse_args()

    '''model'''
    # 如果指定了预训练权重路径，则加载权重
    if args.pretrained_path:
        model = load_pretrained_model(args, args.pretrained_path)
    else:
        model = TimeLLM.Model(args).to(torch.bfloat16)

    '''optimizer'''
    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    '''training_dataloader'''
    train_data, train_loader = data_provider(args, args.data_pretrain, args.data_path_pretrain, True, 'train')
    #vali_data, vali_loader = data_provider(args, args.data_pretrain, args.data_path_pretrain, True, 'val')
    #test_data, test_loader = data_provider(args, args.data, args.data_path, False, 'test')

    '''scheduler'''
    train_steps = len(train_loader)
    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)
    '''Accelerator'''
    device = accelerator.device
    model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        model, model_optim, train_loader, scheduler
    )

    '''training'''
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            print(f"===================Epoch: {epoch} Iter: {iter_count}===================")
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(accelerator.device).to(torch.bfloat16)
            batch_y = batch_y.float().to(accelerator.device).to(torch.bfloat16)
            batch_x_mark = batch_x_mark.float().to(accelerator.device).to(torch.bfloat16)
            batch_y_mark = batch_y_mark.float().to(accelerator.device).to(torch.bfloat16)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                accelerator.device).to(torch.bfloat16)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).to(torch.bfloat16)

            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            accelerator.backward(loss)
            model_optim.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))


if __name__ == '__main__':
    main()
