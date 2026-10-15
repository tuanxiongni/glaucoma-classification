import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, average_precision_score


class FocalLoss(nn.Module):
    """Focal Loss，专门用于缓解类别不平衡。"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class GlaucomaDataset(Dataset):
    """
    青光眼数据集。
    CSV 需要包含 image_name 列；训练/验证集还需要 label 列。
    """
    def __init__(self, image_dir, df, transform=None, is_train=False):
        self.image_dir = image_dir
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_name']
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f'Image not found: {img_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        if 'label' in row:
            label = float(row['label'])
            return image, torch.tensor(label, dtype=torch.float32)
        else:
            return image, img_name


def get_transforms(img_size=380, is_train=True):
    """训练与验证的数据增强。"""
    norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.3),
            norm,
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            norm,
            ToTensorV2()
        ])


def get_weighted_sampler(labels):
    """
    加权随机采样，提升阳性样本（青光眼）在每个 batch 中的出现概率。
    """
    labels = np.asarray(labels, dtype=int)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(labels),
        replacement=True
    )


class AverageMeter:
    """跟踪平均损失。"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(labels, preds):
    """计算 AUROC 与 AUPRC。"""
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    auroc = roc_auc_score(labels, preds)
    auprc = average_precision_score(labels, preds)
    return {'auroc': auroc, 'auprc': auprc}


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_accum=1):
    """训练一个 epoch，支持混合精度与梯度累积。"""
    model.train()
    losses = AverageMeter()
    all_preds, all_labels = [], []

    optimizer.zero_grad()
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, labels) / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        losses.update(loss.item() * grad_accum, images.size(0))
        probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy().flatten())

    metrics = compute_metrics(all_labels, all_preds)
    return losses.avg, metrics


def validate(model, loader, criterion, device):
    """验证，返回平均损失、评估指标以及逐样本预测值与真实标签。"""
    model.eval()
    losses = AverageMeter()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            losses.update(loss.item(), images.size(0))
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy().flatten())

    metrics = compute_metrics(all_labels, all_preds)
    return losses.avg, metrics, all_preds, all_labels
