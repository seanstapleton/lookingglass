import torch
import torch.nn as nn

class BaselineStockPredictor(nn.Module):
    """
    Model that will read in plain stock ticker values over time and decide whether to buy, sell, or hold at the current price.
    """
    def __init__(self, num_series_features=1, num_auxiliary_features=1):
        super().__init__()
        self.recurrent = nn.LSTM(
            input_size=num_series_features,
            hidden_size=128,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
            dropout=0.5
        )
        # concatenate LSTM output with auxiliary features
        # output predicted price
        self.linear = nn.Linear(128+num_auxiliary_features, 1)
        self.init_weights()

    def init_weights(self):
        for layer in [self.recurrent, self.linear]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, X_series, X_auxiliary):
        recurrent_output = self.recurrent(X)
        recurrent_output = torch.mean(recurrent_output, 1)
        
        aux_combined = torch.cat([recurrent_output, X_auxiliary], dim=1)
        output = self.linear(aux_combined)

        return output
