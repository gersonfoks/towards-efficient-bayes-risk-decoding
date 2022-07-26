import torch
from torch.nn import MSELoss

from models.Base.BaseModel import BaseModel


class CometAttentionModel(BaseModel):

    def __init__(self, full_dec_pooling, nmt_down, comet_embedding, nmt_comet_attention, final_layers, initialize_optimizer,
                 device="cuda", ):
        super().__init__()
        self.device_name = device

        self.full_dec_pooling = full_dec_pooling

        self.nmt_down = nmt_down

        self.comet_embedding = comet_embedding

        self.nmt_comet_attention = nmt_comet_attention

        self.final_layers = final_layers



        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

    def forward(self, sources, hypotheses, features):
        nmt_features = self.full_dec_pooling(features)

        query = self.nmt_down(nmt_features)

        comet_features, scores = self.comet_embedding(sources, hypotheses, features["references"])

        comet_pooled = self.nmt_comet_attention(query, comet_features, comet_features)

        final_score = self.final_layers(comet_pooled)

        return final_score
