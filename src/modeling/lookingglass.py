import torch
import torch.nn as nn

class LookingGlassPredictor(nn.Module):
    """
    Model that will use the Baseline predictor as well as earnings call information to decide whether to buy, sell, or hold at the current price
    """
    def __init__(self, num_series_features=1, num_auxiliary_features=1, max_call_len=10000, num_auxiliary_call_features=0):
        """
        Initializes the model.
        Attributes:
            (see baseline.py for num_series_features and num_auxiliary_features)
            max_call_len: maximum number of tokens allowed in an earnings call transcript.
                          We will need to pad each earnings call to be this length (or truncate
                          if the call is too long)
            num_auxiliary_call_features: # non-transcript related features (e.g. if we
                                         include sentiment, ambiguity score, and
                                         confidence score, the num_auxiliary_call_features
                                         would equal 3
        """
        super().__init__()
        self.baseline = BaselineStockPredictor(num_series_features, num_auxiliary_features, 128)
        self.recurrent = nn.LSTM(
            input_size=max_call_len,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.5
        )
        self.rec_linear = nn.Linear(128*2+num_auxiliary_call_features, 128)
        self.combined_linear = nn.Linear(128*2, 128)
        self.final_linear = nn.Linear(128, 1)
    
        self.init_weights()

    def init_weights(self):
        """
        Initialize the model weights
        """
        self.baseline.init_weights()
        for layer in [self.recurrent, self.rec_linear, self.combined_linear, self.final_linear]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, X_series, X_auxiliary, X_transcript, X_transcript_auxiliary):
        """
        Moves the model through each layer
        Parameters:
            (see baseline.py for X_series and X_auxiliary)
            X_transcript: an [N, max_series_features, embedding_size] vector
            X_transcript_auxiliary: an [N, num_auxiliart_features] vector
        """
        baseline_output = self.baseline.forward(X_series, X_auxiliary)
        baseline_activated = nn.functional.relu(baseline_output)

        recurrent_output = self.recurrent(X_transcript)
        recurrent_output = torch.mean(recurrent_output, 1)
        
        aux_combined = torch.cat([recurrent_output, X_transcript_auxiliary], dim=1)
        output = self.rec_linear(aux_combined)
        output_activated = nn.functional.relu(output)

        stock_transcript_joint_layer = torch.cat([baseline_activated, output_activated], dim=1)
        z1 = self.combined_linear(stock_transcript_joint_layer)
        a1 = nn.functional.relu(z1)
        
        final_output = self.final_linear(a1)
        
        return output
        
