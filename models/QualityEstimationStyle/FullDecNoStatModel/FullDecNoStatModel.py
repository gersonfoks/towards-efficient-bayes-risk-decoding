import torch
from torch.nn import MSELoss

from models.base.BaseModel import BaseModel


class FullDecNoStatModel(BaseModel):

    def __init__(self, full_dec_embedding, full_dec_pooling_layers,
                 final_layers, initialize_optimizer, device="cuda", ):
        super().__init__()
        self.device_name = device
        self.full_dec_embedding = full_dec_embedding
        self.full_dec_pooling_layers = full_dec_pooling_layers


        self.final_layers = final_layers

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer
        self.name = 'full_dec_no_stat_model'

    def forward(self, sources, hypotheses, features):
        hidden_layer_embeddings, token_statistic_embedding, attention_mask = self.full_dec_embedding.forward(
            features["input_ids"],
            features["attention_mask"],
            features["decoder_input_ids"],
            features["labels"],
        )

        all_pooled_layers = [

        ]
        for emb, pooling_layer in zip(hidden_layer_embeddings, self.full_dec_pooling_layers):
            all_pooled_layers.append(pooling_layer(emb, attention_mask))


        final_features = torch.concat(all_pooled_layers, dim=-1)
        predicted_scores = self.final_layers(final_features)

        return predicted_scores
