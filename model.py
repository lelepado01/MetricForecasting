import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Predict next value in sequence

    def forward(self, x, lengths):
        # Pack padded sequence
        x = x.unsqueeze(-1)  # Add feature dim: [batch, seq, 1]
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
        packed_out, (hn, cn) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Predict next value at each time step
        predictions = self.fc(out)
        return predictions.squeeze(-1)  # [batch, seq]
    
    def predict_next(self, x):
        # For autoregressive prediction: x shape = [1, seq_len, 1]
        _, (hn, _) = self.lstm(x)
        last_hidden = hn[-1]  # [batch, hidden_dim]
        return self.fc(last_hidden).squeeze(0).squeeze(0)  # scalar