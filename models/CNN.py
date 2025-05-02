import torch
import torch.nn as nn

class CNN2DModel(nn.Module):
    def __init__(self, cnn_channels=64, output_size=120):
        super(CNN2DModel, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=cnn_channels, kernel_size=(5,12), padding=(0,2)
        )
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=(1,15), padding=(0,2)
        )
        
        
        # ---- DO NOT define fc here ----
        self.fc = None  # Temporary placeholder

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = x.squeeze(2)   # remove the feature height (from 1)

        # If fc is not created yet → create dynamically
        if self.fc is None:
            flattened_size = x.shape[1] * x.shape[2]  # channels × width
            self.fc = nn.Linear(flattened_size, 120).to(x.device)

        x = x.flatten(start_dim=1)
        out = self.fc(x)

        return out

