import pytorch_lightning as pl
import torch
from torch.nn import MSELoss


class CometEncodingModel(pl.LightningModule):

    def __init__(self, final_layers, initialize_optimizer, device="cuda", ):
        super().__init__()
        self.device_name = device

        self.final_layers = final_layers

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

    def forward(self, sources, hypotheses, features):

        h_emb = features["hypothesis_embedding"]
        s_emb = features["source_embedding"]



        final_features = [
            h_emb,
            s_emb,
            torch.abs(s_emb - h_emb),
            h_emb * s_emb,


        ]

        for r_emb in features["reference_embeddings"]:
            h_r_prod = h_emb * r_emb
            h_r_diff = torch.abs(h_emb - r_emb)

            final_features += [r_emb, h_r_prod, h_r_diff]


        final_features = torch.concat(final_features, dim=-1)


        predicted_scores = self.final_layers(final_features)

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
