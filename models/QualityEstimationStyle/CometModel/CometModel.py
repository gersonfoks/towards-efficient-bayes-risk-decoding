import torch
from torch.nn import MSELoss

from models.base.BaseModel import BaseModel


class CometModel(BaseModel):

    def __init__(self,  final_layers, initialize_optimizer, device="cuda", ):
        super().__init__()
        self.device_name = device

        self.final_layers = final_layers

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer
        self.name = 'comet_model'

    def forward(self, sources, hypotheses, features):


        s_emb = features["source_embedding"]
        h_emb = features["hypothesis_embedding"]

        feature_vector = torch.concat([
            s_emb, h_emb, s_emb * h_emb, torch.abs(s_emb - h_emb)
        ], dim=-1)

        predicted_scores = self.final_layers(feature_vector)

        return predicted_scores
