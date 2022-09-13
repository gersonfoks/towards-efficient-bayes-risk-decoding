import torch
from torch.nn import MSELoss

from models.base.BaseModel import BaseModel


class FullDecCometModel(BaseModel):

    def __init__(self, full_dec_embedding, full_dec_pooling_layers,
                 token_statistics_pooling_layer, comet_wrapper, final_layers, initialize_optimizer, device="cuda", ):
        super().__init__()
        self.device_name = device

        self.full_dec_embedding = full_dec_embedding
        self.full_dec_pooling_layers = full_dec_pooling_layers
        self.token_statistics_pooling_layer = token_statistics_pooling_layer

        self.comet_wrapper = comet_wrapper
        self.final_layers = final_layers

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer
        self.name = 'full_dec_comet_model'

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

        pooled_statistics = self.token_statistics_pooling_layer(token_statistic_embedding, attention_mask)
        all_pooled_layers.append(pooled_statistics)

        s_emb = self.comet_wrapper.to_embedding(sources)
        h_emb = self.comet_wrapper.to_embedding(hypotheses)

        feature_vector = torch.concat([
            s_emb, h_emb, s_emb * h_emb, torch.abs(s_emb - h_emb)
        ], dim=-1)

        final_features = torch.concat(all_pooled_layers + [feature_vector], dim=-1)
        predicted_scores = self.final_layers(final_features)

        return predicted_scores
