import torch
import torch.nn as nn

class BaselineStockPredictor(nn.Module):
    """
    Model that will read in plain stock ticker values over time and decide whether to buy, sell, or hold at the current price.
    """
    def __init__(self, num_series_features=1, num_auxiliary_features=1, output_size=1):
        """
        Attributes:
            num_series_features: the size of the feature set for an individual
                                 stock price example (e.g. if we include high,
                                 low, average, num_series_features will equal 3
            num_auxiliary_features: the number of auxiliary (not dependent on time)
                                    features we are adding (e.g. if we include the 1yr
                                    high and the market capitalization, num_auxiliary_features
                                    would equal 2
            output_size: the size of the outputted vector. For evaluation, we would use a
                         size of 1 (stock price) or 3 (buy, sell, hold classification).
                         For use in the looking glass model, we want an encoding so we might
                         use a size of 128 to feed into the model.
        """
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
        self.linear = nn.Linear(128+num_auxiliary_features, output_size)
        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights of the model
        """
        for layer in [self.recurrent, self.linear]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, X_series, X_auxiliary):
        """
        Moves the model through each layer
        Parameters:
            X_series: an [N, num_series_examples, num_series_features] size vector
                      where N is the batch size, num_series_examples is how many stock prices
                      we are providing per example (e.g. weekly for the last 3 months), and
                      num_series_features is the same as described in __init__
            X_auxiliary: an [N, num_auxiliary_features] vector
        """
        recurrent_output = self.recurrent(X_series)
        recurrent_output = torch.mean(recurrent_output, 1)
        # We might need this
        # recurrent_output = torch.squeeze(1) 
        aux_combined = torch.cat([recurrent_output, X_auxiliary], dim=1)
        output = self.linear(aux_combined)

        return output
