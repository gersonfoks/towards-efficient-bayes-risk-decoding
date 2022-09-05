import torch
from torch.nn import MSELoss

from models.base.BaseModel import BaseModel


class BasicReferenceModel(BaseModel):

    def __init__(self, full_dec_embedding, full_dec_pooling_layers,
                 token_statistics_pooling_layer, final_layers, initialize_optimizer, device="cuda", min_value=None, max_value=None):
        super().__init__()
        self.device_name = device

        self.full_dec_embedding = full_dec_embedding
        self.full_dec_pooling_layers = full_dec_pooling_layers
        self.token_statistics_pooling_layer = token_statistics_pooling_layer

        self.final_layers = final_layers

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

        self.min_value = min_value
        self.max_value = max_value

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

        pooled_statistics = self.token_statistics_pooling_layer(token_statistic_embedding,attention_mask)
        all_pooled_layers.append(pooled_statistics)

        # Already give a hint what the mean utility is (to better inform how to alter it)
        final_features = torch.concat(all_pooled_layers + [features['mean_utilities'].unsqueeze(dim=-1)], dim=-1)
        predicted_scores_features = self.final_layers(final_features)

        # Learn to alter the predicted scores
        final_score = features['mean_utilities'].unsqueeze(dim=-1).float() + predicted_scores_features


        if self.min_value != None and self.max_value != None:
            self.final_score = torch.clamp(final_score, min=self.min_value, max=self.max_value)

        return final_score
