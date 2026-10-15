import torch
import torch.nn as nn
import torchvision.models as models


class GlaucomaModel(nn.Module):
    """
    青光眼眼底图像二分类模型。
    支持 EfficientNet-B4/B5、ResNet-50、ConvNeXt-Tiny 等骨干网络。
    """
    def __init__(self, backbone='efficientnet_b4', pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = backbone

        if backbone == 'efficientnet_b4':
            weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = models.efficientnet_b4(weights=weights)
            in_features = self.encoder.classifier[1].in_features
            self.encoder.classifier = nn.Identity()
        elif backbone == 'efficientnet_b5':
            weights = models.EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = models.efficientnet_b5(weights=weights)
            in_features = self.encoder.classifier[1].in_features
            self.encoder.classifier = nn.Identity()
        elif backbone == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.encoder = models.resnet50(weights=weights)
            in_features = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        elif backbone == 'convnext_tiny':
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = models.convnext_tiny(weights=weights)
            in_features = self.encoder.classifier[2].in_features
            self.encoder.classifier = nn.Identity()
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2.0),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        feat = self.encoder(x)
        # 处理部分网络返回 tuple 的情况
        if isinstance(feat, tuple):
            feat = feat[0]
        feat = feat.view(feat.size(0), -1)
        out = self.classifier(feat)
        return out
