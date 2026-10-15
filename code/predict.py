import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import GlaucomaModel
from utils import GlaucomaDataset, get_transforms


def predict_with_tta(model, images, tta=False):
    """
    推理，支持简单 TTA：原始 + 水平翻转取平均。
    """
    outputs = torch.sigmoid(model(images))
    if tta:
        outputs_flip = torch.sigmoid(model(torch.flip(images, dims=[3])))
        outputs = (outputs + outputs_flip) / 2.0
    return outputs


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载模型配置
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 构建并加载模型
    model = GlaucomaModel(
        backbone=config['backbone'],
        pretrained=False,
        dropout=config.get('dropout', 0.3)
    ).to(device)
    model_path = os.path.join(args.model_dir, 'best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 读取测试图片列表
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    test_images = [f for f in os.listdir(args.test_dir) if f.lower().endswith(valid_exts)]
    test_images.sort()
    df_test = pd.DataFrame({'image_name': test_images})

    # 测试数据集与 DataLoader
    test_transform = get_transforms(img_size=config['img_size'], is_train=False)
    test_dataset = GlaucomaDataset(args.test_dir, df_test, test_transform, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 推理
    results = []
    with torch.no_grad():
        for images, names in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            probs = predict_with_tta(model, images, tta=args.tta)
            probs = probs.cpu().numpy().flatten()
            for name, prob in zip(names, probs):
                results.append({
                    'Image_name': name,
                    'Probability': float(prob),
                    'Pred_Label': int(prob >= 0.5)
                })

    # 保存提交文件
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    submission = pd.DataFrame(results)
    submission.to_csv(args.output, index=False, encoding='utf-8')
    print(f'Submission saved to: {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Glaucoma Classification Inference')
    parser.add_argument('test_dir', help='测试集图片目录')
    parser.add_argument('output', nargs='?', default='results/submission.csv', help='输出 CSV 路径')
    parser.add_argument('--model_dir', default='model', help='模型目录')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--tta', action='store_true', help='开启测试时增强')
    args = parser.parse_args()
    main(args)
