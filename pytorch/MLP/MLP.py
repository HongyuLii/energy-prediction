import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layers=[256, 256]):
        super().__init__()
        self.input_shape = input_shape   # should be 600 = 120 * 5
        self.output_shape = output_shape  # should be 120 (predictions per timestep)
        self.hidden_layers = hidden_layers
        self.model = self.build_model()

    def build_model(self):
        layers = []
        in_dim = self.input_shape

        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, self.output_shape))
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]           # x: (batch, 120, 5)
        x = x.view(batch_size, -1)        # Flatten to (batch, 600)
        out = self.model(x)               # (batch, 120)
        return out

    def get_model(self):
        return self
