import torch
import torch.nn as nn

class LSTM_CNN_Model(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, cnn_channels=64, output_size=120):
        super(LSTM_CNN_Model, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.conv1d = nn.Conv1d(in_channels=hidden_size,
                                out_channels=cnn_channels,
                                kernel_size=5, 
                                padding=2)  # to keep sequence length the same

        self.relu = nn.ReLU()

        self.fc = nn.Linear(cnn_channels, output_size)  # predict for each time step, or sum if needed

    def forward(self, x):
        # x: (batch_size, sequence_length, num_features)
        lstm_out, _ = self.lstm(x)  # (batch_size, sequence_length, hidden_size)

        # CNN expects (batch_size, channels, sequence_length)
        cnn_in = lstm_out.permute(0, 2, 1)  # swap to (batch, hidden_size, seq_len)
        cnn_out = self.conv1d(cnn_in)  # (batch_size, cnn_channels, sequence_length)
        cnn_out = self.relu(cnn_out)

        # Back to (batch, seq_len, channels)
        cnn_out = cnn_out.permute(0, 2, 1)

        # Apply fully connected to each time step
        out = self.fc(cnn_out)  # (batch_size, sequence_length, output_size)

        # if you just want 1 output per sequence, you can modify here
        return out.squeeze(-1)  # final shape (batch_size, sequence_length)

