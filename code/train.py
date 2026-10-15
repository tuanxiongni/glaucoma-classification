import os
import json
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import GlaucomaModel
from utils import (
    GlaucomaDataset,
    get_transforms,
    get_weighted_sampler,
    FocalLoss,
    train_one_epoch,
    validate
)


def set_seed(seed):
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(args.model_dir, exist_ok=True)

    # 读取训练与验证标签
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    # 计算阳性样本权重
    pos = train_df['label'].sum()
    neg = len(train_df) - pos
    pos_weight = neg / (pos + 1e-6)
    print(f'Train samples: {len(train_df)}, Pos: {pos}, Neg: {neg}, Pos weight: {pos_weight:.4f}')

    # 保存模型配置，供 predict.py 使用
    config = {
        'backbone': args.backbone,
        'img_size': args.img_size,
        'dropout': args.dropout,
    }
    with open(os.path.join(args.model_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    # 数据增强与 DataLoader
    train_transform = get_transforms(img_size=args.img_size, is_train=True)
    val_transform = get_transforms(img_size=args.img_size, is_train=False)

    train_dataset = GlaucomaDataset(args.train_dir, train_df, train_transform, is_train=True)
    val_dataset = GlaucomaDataset(args.val_dir, val_df, val_transform, is_train=False)

    sampler = get_weighted_sampler(train_df['label'].values)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 模型
    model = GlaucomaModel(
        backbone=args.backbone,
        pretrained=True,
        dropout=args.dropout
    ).to(device)

    # 损失函数：默认 Focal Loss，可选带 pos_weight 的 BCE
    if args.loss == 'focal':
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        weight = torch.tensor([pos_weight], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

    # 优化器与学习率调度
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if (not args.no_amp and torch.cuda.is_available()) else None

    # 训练循环
    best_auroc = 0.0
    val_names = list(val_df['image_name'])
    history = []
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, grad_accum=args.grad_accum
        )
        val_loss, val_metrics, val_preds, val_labels = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f'Train  Loss: {train_loss:.4f} | AUROC: {train_metrics["auroc"]:.4f} | AUPRC: {train_metrics["auprc"]:.4f}')
        print(f'Val    Loss: {val_loss:.4f} | AUROC: {val_metrics["auroc"]:.4f} | AUPRC: {val_metrics["auprc"]:.4f}')
        print(f'LR: {lr:.2e}')

        # 记录逐 epoch 历史指标
        history.append({
            'epoch': epoch + 1,
            'lr': lr,
            'train_loss': train_loss,
            'train_auroc': train_metrics['auroc'],
            'train_auprc': train_metrics['auprc'],
            'val_loss': val_loss,
            'val_auroc': val_metrics['auroc'],
            'val_auprc': val_metrics['auprc'],
        })
        pd.DataFrame(history).to_csv(
            os.path.join(args.model_dir, 'history.csv'), index=False
        )

        # 记录验证集逐样本预测结果
        val_records = pd.DataFrame({
            'image_name': val_names,
            'label': [int(x) for x in val_labels],
            'prob': [float(x) for x in val_preds],
        })
        val_records['pred'] = (val_records['prob'] >= 0.5).astype(int)
        val_records.to_csv(
            os.path.join(args.model_dir, 'val_predictions.csv'), index=False
        )

        # 保存验证 AUROC 最高的模型，并额外保存该 epoch 的验证集记录
        if val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
            val_records.to_csv(
                os.path.join(args.model_dir, 'val_predictions_best.csv'), index=False
            )
            print(f'>> Saved best model, AUROC = {best_auroc:.4f}')

    print(f'\nTraining complete. Best AUROC: {best_auroc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Glaucoma Classification Training')
    parser.add_argument('--train_csv', default='data/train.csv', help='训练集标签 CSV')
    parser.add_argument('--val_csv', default='data/val.csv', help='验证集标签 CSV')
    parser.add_argument('--train_dir', default='data/train', help='训练集图片目录')
    parser.add_argument('--val_dir', default='data/val', help='验证集图片目录')
    parser.add_argument('--model_dir', default='model', help='模型保存目录')
    parser.add_argument('--backbone', default='efficientnet_b4', help='骨干网络')
    parser.add_argument('--img_size', type=int, default=380, help='输入尺寸')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout 概率')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--loss', choices=['bce', 'focal'], default='focal', help='损失函数')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='Focal alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal gamma')
    parser.add_argument('--no_amp', action='store_true', help='关闭混合精度训练')
    parser.add_argument('--grad_accum', type=int, default=1, help='梯度累积步数')
    args = parser.parse_args()
    main(args)
