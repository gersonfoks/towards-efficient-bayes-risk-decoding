import pytorch_lightning as pl
import torch
from torch.nn import MSELoss

from models.Base.BaseModel import BaseModel


class CrossAttentionModelV4(BaseModel):

    def __init__(self, embedding_layer, prob_and_entr_embedding_layer, self_attention_layer, cross_attention_layer,
                 weights_layer, h_pred_layer,ref_pred_layer,
                 initialize_optimizer,
                 n_references=3,
                 device="cuda", ):
        super().__init__()
        self.device_name = device

        self.embedding_layer = embedding_layer

        self.prob_entr_embedding_layer = prob_and_entr_embedding_layer

        self.cross_attention_layer = cross_attention_layer
        self.self_attention_layer = self_attention_layer

        self.weights_layer = weights_layer
        self.ref_pred_layer = ref_pred_layer
        self.h_pred_layer = h_pred_layer

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer
        self.n_references = n_references

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, sources, hypotheses, features):
        hidden_states = []
        # Embed the hypothesis
        h_emb = self.embedding_layer(features["tokenized_hypotheses"]["input_ids"])
        h_pe_emb = self.prob_entr_embedding_layer(features["hypotheses_prob_entropy"].float())

        h_attention_mask = features["tokenized_hypotheses"]["attention_mask"]

        h_final = self.self_attention_layer(h_emb, h_emb, h_pe_emb, h_attention_mask, h_attention_mask).squeeze(dim=1)

        predictions = []
        h_pred = self.h_pred_layer(h_final)
        predictions.append(h_pred)

        hidden_states.append(h_final)

        ref_preds = []

        ### Forward the references
        for r_tokenized, r_prob_entr in zip(features["references_tokenized"], features["reference_prob_entropy"]):
            r_emb = self.embedding_layer(r_tokenized["input_ids"])
            r_pe_emb = self.prob_entr_embedding_layer(r_prob_entr.float())
            ref_attention_mask = r_tokenized["attention_mask"].bool()

            r_final = self.cross_attention_layer(h_emb, r_emb, r_pe_emb, h_attention_mask, ref_attention_mask).squeeze(dim=1)
            predictions.append(self.ref_pred_layer(r_final))

            hidden_states.append(r_final)

        # Next we concat the hidden_states and make features out of it.


        hidden_states = torch.concat(hidden_states, dim=-1)

        weights = self.softmax(self.final_layers(hidden_states))

        #Next we take a weighted sum

        predictions = torch.concat(predictions)
        predicted_scores = torch.sum(predictions * weights, dim=0)




        return predicted_scores
