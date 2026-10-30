"""Small randomly initialised architectures with comparable parameter counts."""

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        previous = 3
        for width in (20, 40, 80, 160):
            blocks.extend([nn.Conv2d(previous, width, 3, padding=1, bias=False),
                           nn.GroupNorm(4, width), nn.GELU(), nn.MaxPool2d(2)])
            previous = width
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(160, 3)

    def forward(self, x):
        return self.head(self.features(x))


class SmallViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 64, kernel_size=8, stride=8)
        self.cls = nn.Parameter(torch.zeros(1, 1, 64))
        self.position = nn.Parameter(torch.empty(1, 65, 64))
        nn.init.trunc_normal_(self.position, std=.02)
        block = nn.TransformerEncoderLayer(64, 4, dim_feedforward=128, dropout=0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, 4, enable_nested_tensor=False)
        # TransformerEncoder copies its input layer; initialise each copy separately.
        for layer in self.encoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.xavier_uniform_(layer.linear2.weight)
        self.norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, 3)

    def forward(self, x):
        patches = self.patch(x).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls.expand(x.shape[0], -1, -1), patches], dim=1)
        return self.head(self.norm(self.encoder(tokens + self.position)[:, 0]))


def build_model(architecture):
    if architecture == "cnn":
        return SmallCNN()
    if architecture == "vit":
        return SmallViT()
    raise ValueError("Architecture must be cnn or vit")


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
