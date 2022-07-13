import pytorch_lightning as pl
import torch
from torch.nn import MSELoss


class BasicCrossAttentionModel(pl.LightningModule):

    def __init__(self, embedding_layer, cross_attention_layer, pooling_layer, final_layers, initialize_optimizer,
                 device="cuda", ):
        super().__init__()
        self.device_name = device

        self.embedding_layer = embedding_layer

        self.cross_attention_layer = cross_attention_layer

        self.pooling_layer = pooling_layer

        self.final_layers = final_layers

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

    def forward(self, sources, hypotheses, features):
        # Embed the hypothesis
        embedded_hypotheses = self.embedding_layer(features["tokenized_hypotheses"]["input_ids"])

        hypotheses_mask = features["tokenized_hypotheses"]["attention_mask"]


        # Map the references to embedding space
        hidden_states = []

        hidden_state, _ = self.cross_attention_layer(query=embedded_hypotheses, key=embedded_hypotheses,
                                                     value=embedded_hypotheses,
                                                     key_padding_mask=~hypotheses_mask.bool(),
                                                     )

        hidden_states.append(self.pooling_layer(hidden_state, ~hypotheses_mask.bool()))

        for references in features["tokenized_references"]:
            embedded_reference = self.embedding_layer(references["input_ids"])
            ref_attention_mask = ~references["attention_mask"].bool()
            hidden_state, _ = self.cross_attention_layer(query=embedded_hypotheses, key=embedded_reference,
                                                      value=embedded_reference,
                                                      key_padding_mask=ref_attention_mask,
                                                      )


            hidden_states.append(self.pooling_layer(hidden_state, ~hypotheses_mask.bool()))

        # Next we concat the hidden_states

        features = torch.concat(hidden_states, dim=-1)

        predicted_scores = self.final_layers(features)

        return predicted_scores

    def batch_to_out(self, batch):

        sources, hypotheses, features, scores = batch
        predicted_scores = self.forward(sources, hypotheses, features).flatten()

        loss = self.criterion(predicted_scores, scores.to(self.device))

        return {"loss": loss}

    def training_step(self, batch, batch_idx):

        batch_out = self.batch_to_out(batch)

        sources, hypotheses, features, scores = batch

        batch_size = len(scores)

        loss = batch_out["loss"]

        for log_var in self.log_vars:
            self.log("train_{}".format(log_var), batch_out[log_var], batch_size=batch_size)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        batch_out = self.batch_to_out(batch)

        sources, hypotheses, features, scores = batch

        batch_size = len(scores)

        for log_var in self.log_vars:
            self.log("val_{}".format(log_var), batch_out[log_var], batch_size=batch_size)

    def configure_optimizers(self):

        return self.initialize_optimizer(self.parameters())
