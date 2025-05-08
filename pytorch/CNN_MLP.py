import torch
import torch.nn as nn

class CNN2D_MLP_Model(nn.Module):
    def __init__(self, cnn_channels=64, hidden_layers=[512, 512, 512]):
        super(CNN2D_MLP_Model, self).__init__()

        self.conv1 = nn.Conv2d(1, cnn_channels, kernel_size=(5, 3), padding=(0, 1))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(cnn_channels, cnn_channels, kernel_size=(1, 3), padding=(0, 1))

        # Build MLP: cnn_channels → 512 → 512 → 512 → 1
        layers = []
        in_dim = cnn_channels
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))  # output 1 value per timestep
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):  # x: (batch, 120, 5)
        x = x.permute(0, 2, 1)      # → (batch, 5, 120)
        x = x.unsqueeze(1)          # → (batch, 1, 5, 120) → (B, C_in=1, H=features, W=time)
        
        x = self.relu(self.conv1(x))  # kernel=(5, 3) → over all features and small time window
        x = self.relu(self.conv2(x))

        #print("After conv2:", x.shape)

        x = torch.mean(x, dim=2)       # avg over features → (batch, channels, time)
        x = x.permute(0, 2, 1)         # (batch, time, channels)
        out = self.mlp(x)              # (batch, time, 1)
        return out.squeeze(-1)         # (batch, time)
