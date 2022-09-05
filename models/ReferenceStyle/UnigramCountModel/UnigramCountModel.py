import torch
from torch.nn import MSELoss

from models.base.BaseModel import BaseModel


class UnigramCountModel(BaseModel):

    def __init__(self, unigram_count_embedding, attention_layer, final_layers, initialize_optimizer, device="cuda",  min_value=None, max_value=None):
        super().__init__()
        self.device_name = device

        self.unigram_count_embedding = unigram_count_embedding

        self.attention_layer = attention_layer

        self.final_layers = final_layers

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

        self.min_value = min_value
        self.max_value = max_value

    def forward(self, sources, hypotheses, features):
        hypotheses_embedding = self.unigram_count_embedding(features["hypothesis_ids"])

        references_embeddings = []
        for reference_ids in features["references_ids"]:
            references_embeddings.append(self.unigram_count_embedding(
                reference_ids
            ))

        # Transform the reference embeddings
        references_embeddings = torch.stack(references_embeddings)



        hypotheses_embedding = hypotheses_embedding.unsqueeze(dim=1)

        att_out, _ = self.attention_layer(key=references_embeddings, value=references_embeddings, query=hypotheses_embedding)

        att_out = att_out.squeeze(dim=1)

        # Already give a hint what the mean utility is (to better inform how to alter it)
        final_features = torch.concat([att_out, features['mean_utilities'].unsqueeze(dim=-1)], dim=-1)
        predicted_scores_features = self.final_layers(final_features)

        # Learn to alter the predicted scores
        final_score = features['mean_utilities'].unsqueeze(dim=-1).float() + predicted_scores_features

        if self.min_value != None and self.max_value != None:
            self.final_score = torch.clamp(final_score, min=self.min_value, max=self.max_value)


        return final_score
