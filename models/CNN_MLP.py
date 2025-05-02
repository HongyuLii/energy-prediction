import torch
import torch.nn as nn

class CNN2D_MLP_Model(nn.Module):
    def __init__(self, cnn_channels=64):
        super(CNN2D_MLP_Model, self).__init__()

        self.conv1 = nn.Conv2d(1, cnn_channels, kernel_size=(5,15), padding=(0,2))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(cnn_channels, cnn_channels, kernel_size=(1,5), padding=(0,2))

        self.mlp = nn.Sequential(
            nn.Linear(cnn_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = x.squeeze(2)         # (batch_size, cnn_channels, 120)
        x = x.permute(0, 2, 1)   # (batch_size, 120, cnn_channels)

        out = self.mlp(x)        # (batch_size, 120, 1)
        out = out.squeeze(-1)    # (batch_size, 120)

        return out
