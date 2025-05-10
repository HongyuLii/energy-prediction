import torch
import torch.nn as nn

class CNN2D_Model(nn.Module):
    """
    Input : (B, 168, 2)            # 7 days × 2 features
    Output: (B, 24)                # next‑day 24‑step forecast
    """
    def __init__(self, n_features=2, horizon=24, cnn_channels=64):
        super().__init__()

        # 2‑D convolutions over (features × time)
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, cnn_channels,
                      kernel_size=(n_features, 3),
                      padding=(0, 1)),          # keep time length
            nn.ReLU(inplace=True),

            nn.Conv2d(cnn_channels, cnn_channels,
                      kernel_size=(1, 5),
                      padding=(0, 2)),
            nn.ReLU(inplace=True),
        )

        # compress 168 → 24 on the time axis
        self.time_pool = nn.AdaptiveAvgPool1d(horizon)   # (B, C, 24)

        # point‑wise 1×1 Conv to map C → 1 at every time step
        self.head = nn.Conv1d(cnn_channels, 1, kernel_size=1)  # (B, 1, 24)

    def forward(self, x):                # x: (B, 168, 2)
        x = x.permute(0, 2, 1).unsqueeze(1)   # (B, 1, 2, 168)

        x = self.conv_block(x)                # (B, C, 1, 168)
        x = x.squeeze(2)                      # (B, C, 168)

        x = self.time_pool(x)                 # (B, C, 24)
        x = self.head(x)                      # (B, 1, 24)

        return x.squeeze(1)                   # (B, 24)
