import pytorch_lightning as pl
import torch
from torch.nn import MSELoss

from models.Base.BaseModel import BaseModel


class ProbSumModel(BaseModel):

    def __init__(self, lstm_layer,   final_layers, initialize_optimizer,
                 device="cuda", ):
        super().__init__()
        self.device_name = device

        self.lstm_layer = lstm_layer




        self.final_layers = final_layers

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

    def forward(self, sources, hypotheses, features):


        # Embed the hypothesis

        _, (h_p_n, _) = self.lstm_layer(features["probs"])



        final_features =  h_p_n.reshape(-1, 512)

        predicted_scores = self.final_layers(final_features)

        return predicted_scores
