import pytorch_lightning as pl
import torch
from torch.nn import MSELoss

from models.Base.BaseModel import BaseModel


class CrossAttentionModelV2(BaseModel):

    def __init__(self, embedding_layer, prob_and_entr_embedding_layer, learnable_embedding,
                 learnable_embedding_prob_entropy, cross_attention_layer, pooling_layer, final_layers,
                 initialize_optimizer,
                 device="cuda", ):
        super().__init__()
        self.device_name = device

        self.embedding_layer = embedding_layer

        self.prob_entr_embedding_layer = prob_and_entr_embedding_layer

        self.cross_attention_layer = cross_attention_layer

        self.pooling_layer = pooling_layer

        self.final_layers = final_layers

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

        self.learnable_embedding = learnable_embedding
        self.learnable_embedding_prob_entropy = learnable_embedding_prob_entropy

    def forward(self, sources, hypotheses, features):
        # Embed the hypothesis

        h_emb = self.embedding_layer(features["tokenized_hypotheses"]["input_ids"])
        batch_size = h_emb.shape[0]
        learnable_emb = self.learnable_embedding.repeat(batch_size, 1, 1)

        h_emb_extended = torch.concat([learnable_emb, h_emb], dim=1)

        h_pe_emb = self.prob_entr_embedding_layer(features["hypotheses_prob_entropy"].float())

        learnable_emb_prob_entropy = self.learnable_embedding_prob_entropy.repeat(batch_size, 1, 1)

        # Concat learnable param to sequence
        h_pe_emb = torch.concat([learnable_emb_prob_entropy, h_pe_emb], dim=1)

        hypotheses_mask = features["tokenized_hypotheses"]["attention_mask"]

        extended_hypotheses_mask = torch.concat([torch.ones(batch_size, 2).to("cuda"), hypotheses_mask], dim=-1)

        # Map the references to embedding space
        hidden_states = []

        hidden_state, _ = self.cross_attention_layer(query=h_emb, key=h_emb_extended,
                                                     value=h_pe_emb,
                                                     key_padding_mask=~extended_hypotheses_mask.bool(),
                                                     )

        hidden_states.append(self.pooling_layer(hidden_state, hypotheses_mask.bool()))

        ### Forward the references
        for r_tokenized, r_prob_entr in zip(features["references_tokenized"], features["reference_prob_entropy"]):
            r_emb = self.embedding_layer(r_tokenized["input_ids"])

            r_emb_extended = torch.concat([learnable_emb, r_emb], dim=1)

            r_pe_emb = self.prob_entr_embedding_layer(r_prob_entr.float())

            r_pe_emb_extended = torch.concat([learnable_emb_prob_entropy, r_pe_emb], dim=1)

            ref_attention_mask = r_tokenized["attention_mask"].bool()
            ref_attention_mask = torch.concat([torch.ones(batch_size, 2).to("cuda"), ref_attention_mask], dim=-1)

            hidden_state, _ = self.cross_attention_layer(query=h_emb, key=r_emb_extended,
                                                         value=r_pe_emb_extended,
                                                         key_padding_mask=~ref_attention_mask.bool(),
                                                         )

            hidden_states.append(self.pooling_layer(hidden_state, hypotheses_mask.bool()))

        # Next we concat the hidden_states

        features = torch.concat(hidden_states, dim=-1)

        predicted_scores = self.final_layers(features)

        return predicted_scores

    # def parameters(self, recurse: bool = True):
    #     return [self.prob_entr_embedding_layer.parameters()] + [self.cross_attention_layer.parameters()] + [self.final_layers.parameters()]
