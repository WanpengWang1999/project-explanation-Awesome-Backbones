import os
import sys
# from typing import Sequence
sys.path.insert(0, os.getcwd())
import copy
import argparse     # 命令行解析
import shutil       # os模块的补充，提供了复制、移动、删除、压缩、解压等操作
import time
import numpy as np
import random

import torch
# import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

from utils.history import History
from utils.dataloader import Mydataset, collate
from utils.train_utils import train, validation, print_info, file2dict, init_random_seed, set_random_seed, resume_model
from utils.inference import init_model
from core.optimizers import *
from models.build import BuildNet

'''
参数说明：
--config：模型配置文件路径
--resume-from：是否从断点开始训练，指定权重路径（.pth文件）
--seed：是否指定随机种子
--device：指定GPU或者CPU进行训练
--split-validation：是否从训练集中重新划分验证集
--ratio：重新划分验证集所占的比例
--deterministic：是否使用确定性的CUDNN backend，多GPU训练有关
--local-rank：指定训练GPU序号，多GPU训练有关
'''
def parse_args():   # 命令行解析操作，
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', default='models/mobilenet/mobilenet_v2_.py', help='train config file path')
    # 增加config默认值，便于直接运行
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--device', help='device used for training. (Deprecated)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--split-validation',
        action='store_true',
        help='whether to split validation set from training set.')
    parser.add_argument(
        '--ratio',
        type=float,
        default=0.2,
        help='the proportion of the validation set to the training set.')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    # 读取配置文件获取关键字段
    args = parse_args()
    model_cfg, train_pipeline, val_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)
    # 读取配置文件，返回文件中对应内容，保留原始格式
    print_info(model_cfg)   # 使用表格对配置文件中模型信息进行打印

    # 初始化
    meta = dict()   # 用于存储历史信息
    dirname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())  # 生成模型存储文件夹名
    save_dir = os.path.join('logs', model_cfg.get('backbone').get('type'), dirname)
    meta['save_dir'] = save_dir
    
    # 设置随机数种子
    seed = init_random_seed(args.seed)      # 不设定，返回seed = np.random.randint(2**31)
    set_random_seed(seed, deterministic=args.deterministic)     # 对使用随机数的地方设定固定随机数种子
    meta['seed'] = seed
    
    # 读取训练&制作验证标签数据
    total_annotations   = "datas/train.txt"
    with open(total_annotations, encoding='utf-8') as f:
        total_datas = f.readlines()
    if args.split_validation:               # 重新划分的验证集
        total_nums = len(total_datas)
        # indices = list(range(total_nums))
        if isinstance(seed, int):
            rng = np.random.default_rng(seed)       # 使用指定的随机数种子创建随机数生成器
            rng.shuffle(total_datas)                # 打乱总数据集顺序
        val_nums = int(total_nums * args.ratio)
        folds = list(range(int(1.0 / args.ratio)))  # 计算总数量（份数）
        fold = random.choice(folds)                 # 随机取一份
        val_start = val_nums * fold
        val_end = val_nums * (fold + 1)
        train_datas = total_datas[:val_start] + total_datas[val_end:]
        val_datas = total_datas[val_start:val_end]
    else:
        train_datas = total_datas.copy()
        test_annotations    = 'datas/test.txt'
        with open(test_annotations, encoding='utf-8') as f:
            val_datas   = f.readlines()
    
    # 初始化模型,详见https://www.bilibili.com/video/BV12a411772h
    if args.device is not None:     # 选择训练设备
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Initialize the weights.')

    model = BuildNet(model_cfg)     # 模型初始化

    if not data_cfg.get('train').get('pretrained_flag'):
        model.init_weights()
    if data_cfg.get('train').get('freeze_flag') and data_cfg.get('train').get('freeze_layers'):
        freeze_layers = ' '.join(list(data_cfg.get('train').get('freeze_layers')))
        print('Freeze layers : ' + freeze_layers)
        model.freeze_layers(data_cfg.get('train').get('freeze_layers'))
    
    if device != torch.device('cpu'):
        model = DataParallel(model, device_ids=[args.gpu_id])
    
    # 初始化优化器
    optimizer = eval('optim.' + optimizer_cfg.pop('type'))(params=model.parameters(), **optimizer_cfg)
    
    # 初始化学习率更新策略
    lr_update_func = eval(lr_config.pop('type'))(**lr_config)
    
    # 制作数据集->数据增强&预处理,详见https://www.bilibili.com/video/BV1zY4y167Ju
    train_dataset = Mydataset(train_datas, train_pipeline)
    val_pipeline = copy.deepcopy(train_pipeline)
    val_dataset = Mydataset(val_datas, val_pipeline)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=data_cfg.get('batch_size'), num_workers=data_cfg.get('num_workers'),pin_memory=True, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=data_cfg.get('batch_size'), num_workers=data_cfg.get('num_workers'), pin_memory=True,
    drop_last=True, collate_fn=collate)
    
    # 将关键字段存储，方便训练时同步调用&更新
    runner = dict(
        optimizer         = optimizer,
        train_loader      = train_loader,
        val_loader        = val_loader,
        iter              = 0,
        epoch             = 0,
        max_epochs       = data_cfg.get('train').get('epoches'),
        max_iters         = data_cfg.get('train').get('epoches')*len(train_loader),
        best_train_loss   = float('INF'),
        best_val_acc     = float(0),
        best_train_weight = '',
        best_val_weight   = '',
        last_weight       = ''
    )
    meta['train_info'] = dict(train_loss = [],
                              val_loss = [],
                              train_acc = [],
                              val_acc = [])
    
    # 是否从中断处恢复训练
    if args.resume_from:
        model, runner, meta = resume_model(model, runner, args.resume_from, meta)
    else:
        os.makedirs(save_dir)
        shutil.copyfile(args.config,os.path.join(save_dir,os.path.split(args.config)[1]))
        model = init_model(model, data_cfg, device=device, mode='train')
        
    # 初始化保存训练信息类
    train_history =History(meta['save_dir'])
    
    # 记录初始学习率，详见https://www.bilibili.com/video/BV1WT4y1q7qN
    lr_update_func.before_run(runner)
    
    # 训练
    for epoch in range(runner.get('epoch'),runner.get('max_epochs')):
        lr_update_func.before_train_epoch(runner)
        train(model, runner, lr_update_func, device, epoch, data_cfg.get('train').get('epoches'), data_cfg.get('test'), meta)
        validation(model, runner, data_cfg.get('test'), device, epoch, data_cfg.get('train').get('epoches'), meta)
        
        train_history.after_epoch(meta)

if __name__ == "__main__":
    main()
