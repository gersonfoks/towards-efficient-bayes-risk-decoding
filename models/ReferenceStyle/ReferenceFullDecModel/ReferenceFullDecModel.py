import torch
from torch.nn import MSELoss

from models.Base.BaseModel import BaseModel


class ReferenceFullDecModel(BaseModel):

    def __init__(self, full_dec_embedding, token_statistics_embedding, full_dec_pooling_layers, token_statistics_pooling_layer, final_layers, gate, initialize_optimizer, device="cuda", ):
        super().__init__()
        self.device_name = device


        self.token_statistics_embedding = token_statistics_embedding
        self.full_dec_embedding = full_dec_embedding
        self.full_dec_pooling_layers = full_dec_pooling_layers
        self.token_statistics_pooling_layer = token_statistics_pooling_layer


        self.final_layers = final_layers

        self.gate = gate

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

    def forward(self, sources, hypotheses, features):

        embeddings, attention_mask = self.full_dec_embedding.forward(features["input_ids"],
                                                                              features["attention_mask"],
                                                                              features["decoder_input_ids"],
                                                                              features["labels"],
                                                                              )


        all_pooled_layers = [

        ]
        for emb, pooling_layer in zip(embeddings, self.full_dec_pooling_layers):
            all_pooled_layers.append(pooling_layer(emb, attention_mask))

        statistics_embeddings = self.token_statistics_embedding.forward(features["token_statistics"], )

        pooled_statistics = self.token_statistics_pooling_layer(statistics_embeddings, features["attention_mask_tokens"])
        all_pooled_layers.append(pooled_statistics)

        final_features = torch.concat(all_pooled_layers, dim=-1)
        predicted_scores_features = self.final_layers(final_features)

        gate = torch.sigmoid(self.gate)

        final_score = (gate * predicted_scores_features + (1 - gate) * features['mean_utilities'].unsqueeze(dim=-1)).float()




        return final_score





