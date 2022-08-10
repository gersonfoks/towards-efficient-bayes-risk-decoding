import torch
from torch.nn import MSELoss
from torch.nn.utils.rnn import pack_sequence

from models.base.BaseModel import BaseModel


class BasicLstmModel(BaseModel):

    def __init__(self, source_embedding, hypothesis_embedding, lstm_layer, final_layers, initialize_optimizer,
                 device="cuda", ):
        super().__init__()
        self.device_name = device

        self.source_embedding = source_embedding
        self.hypothesis_embedding = hypothesis_embedding

        self.lstm_layer = lstm_layer

        self.final_layers = final_layers

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

    def forward(self, sources, hypotheses, features):
        s_emb = self.source_embedding.forward(features["tokenized_source"])
        h_emb = self.source_embedding.forward(features["tokenized_hypothesis"])

        s_att_mask = features["source_attention_mask"]
        h_att_mask = features["hypothesis_attention_mask"]
        s_h_concat = self.concat_and_pack(s_emb, h_emb, s_att_mask, h_att_mask)

        _, (h_n, _) = self.lstm_layer(s_h_concat)

        h_n = h_n.permute(1, 0, 2).reshape(-1, h_n.shape[-1] * 2)

        predicted_scores = self.final_layers(h_n)

        return predicted_scores

    def concat_and_pack(self, s_emb, h_emb, s_att_mask, h_att_mask):
        result = []

        for s, h, s_mask, h_mask in zip(s_emb, h_emb, s_att_mask, h_att_mask):
            s_index = s_mask.sum()
            h_index = h_mask.sum()

            concatonated = torch.concat([s[:s_index], h[:h_index]])

            result.append(
                concatonated
            )

        return pack_sequence(result, enforce_sorted=False)
