from torch.nn import MSELoss

from models.base.BaseModel import BaseModel


class TokenStatisticsModel(BaseModel):

    def __init__(self, token_statistics_embedding, pooling_layer, final_layers, initialize_optimizer, device="cuda", ):
        super().__init__()
        self.device_name = device

        self.token_statistics_embedding = token_statistics_embedding
        self.pooling_layer = pooling_layer

        self.final_layers = final_layers

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

    def forward(self, sources, hypotheses, features):
        embeddings, attention_mask = self.token_statistics_embedding.forward(features["input_ids"],
                                                                             features["attention_mask"],
                                                                             features["decoder_input_ids"],
                                                                             features["labels"],
                                                                             )
        pooled_embedding = self.pooling_layer(embeddings, attention_mask)

        predicted_scores = self.final_layers(pooled_embedding)

        return predicted_scores
