import pytorch_lightning as pl
import torch
from torch.nn import MSELoss

from models.Base.BaseModel import BaseModel


class ProbSumModelV2(BaseModel):

    def __init__(self, gru_h_layer,gru_top_n_layer , final_layers, initialize_optimizer,
                 device="cuda", ):
        super().__init__()
        self.device_name = device

        self.gru_h_layer = gru_h_layer
        self.gru_top_n_layer = gru_top_n_layer

        self.final_layers = final_layers

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

    def forward(self, sources, hypotheses, features):
        # Preprocessing:

        _, h_probs_hidden = self.gru_h_layer(features["h_probs"])
        _, top_k_hidden = self.gru_top_n_layer(features["top_k_probs"])
        final_features = torch.concat([
            h_probs_hidden.permute(1, 0, 2).reshape(-1, 128),
            top_k_hidden.permute(1, 0, 2).reshape(-1, 512),
        ], dim=-1)

        predicted_scores = self.final_layers(final_features)

        return predicted_scores
