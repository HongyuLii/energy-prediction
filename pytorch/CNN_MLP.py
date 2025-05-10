import torch
import torch.nn as nn


class CNN2D_MLP_Model(nn.Module):
    def __init__(self, n_features=2, horizon=24, cnn_channels=64,
                 hidden_dims=(512, 512, 512)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=(n_features, 3), padding=(0,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=(1,5), padding=(0,2)),
            nn.ReLU(inplace=True),
        )
        self.time_pool = nn.AdaptiveAvgPool1d(horizon)   # 168 → 24
        mlp = []
        in_dim = cnn_channels
        for h in hidden_dims:
            mlp += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
            in_dim = h
        mlp.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):            # x:(B,168,2)
        x = x.permute(0,2,1).unsqueeze(1)  # (B,1,2,168)
        x = self.conv(x)                   # (B,C,1,168)
        x = x.squeeze(2)                   # (B,C,168)
        x = self.time_pool(x)              # (B,C,24)
        x = x.permute(0,2,1)               # (B,24,C)
        return self.mlp(x).squeeze(-1)     # (B,24)
