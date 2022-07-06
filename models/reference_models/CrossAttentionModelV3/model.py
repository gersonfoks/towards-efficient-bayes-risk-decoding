import pytorch_lightning as pl
import torch
from torch.nn import MSELoss

from models.Base.BaseModel import BaseModel


class CrossAttentionModelV3(BaseModel):

    def __init__(self, embedding_layer, prob_and_entr_embedding_layer, self_attention_layer, cross_attention_layer,
                 final_layers,
                 initialize_optimizer,
                 device="cuda", ):
        super().__init__()
        self.device_name = device

        self.embedding_layer = embedding_layer

        self.prob_entr_embedding_layer = prob_and_entr_embedding_layer

        self.cross_attention_layer = cross_attention_layer
        self.self_attention_layer = self_attention_layer

        self.final_layers = final_layers

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

    def forward(self, sources, hypotheses, features):
        hidden_states = []
        # Embed the hypothesis
        h_emb = self.embedding_layer(features["tokenized_hypotheses"]["input_ids"])
        h_pe_emb = self.prob_entr_embedding_layer(features["hypotheses_prob_entropy"].float())

        h_attention_mask = features["tokenized_hypotheses"]["attention_mask"]

        h_final = self.self_attention_layer(h_emb, h_emb, h_pe_emb, h_attention_mask, h_attention_mask)

        hidden_states.append(h_final)

        ### Forward the references
        for r_tokenized, r_prob_entr in zip(features["references_tokenized"], features["reference_prob_entropy"]):
            r_emb = self.embedding_layer(r_tokenized["input_ids"])
            r_pe_emb = self.prob_entr_embedding_layer(r_prob_entr.float())
            ref_attention_mask = r_tokenized["attention_mask"].bool()

            r_final = self.cross_attention_layer(h_emb, r_emb, r_pe_emb, h_attention_mask, ref_attention_mask)

            hidden_states.append(r_final)

        # Next we concat the hidden_states and make features out of it.


        features = torch.concat(hidden_states, dim=-1).squeeze(dim=1)

        predicted_scores = self.final_layers(features)

        return predicted_scores
