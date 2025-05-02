import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layers=[256, 256]):
        super().__init__()
        self.input_shape = input_shape   # should be 5
        self.output_shape = output_shape  # should be 1
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
        batch_size, time_steps, features = x.shape  # (batch, 120, 5)
        x = x.view(-1, features)                    # (batch*120, 5)
        print(x.shape)
        out = self.model(x)                         # (batch*120, 1)
        print(out.shape)
        out = out.reshape(batch_size, time_steps, self.output_shape)  # (batch, 120, output_size)
        return out.squeeze(-1)  # (batch, 120)

    def get_model(self):
        return self
