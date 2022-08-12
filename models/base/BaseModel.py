from time import time

import pytorch_lightning as pl
import torch


class BaseModel(pl.LightningModule):

    def batch_to_out(self, batch):

        sources, hypotheses, features, scores = batch
        predicted_scores = self.forward(sources, hypotheses, features).flatten()

        loss = self.criterion(predicted_scores, scores.to(self.device))

        return {"loss": loss, "predictions": predicted_scores}

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

    @torch.no_grad()
    def predict(self, batch):

        batch_out = self.batch_to_out(batch)

        return batch_out


    @torch.no_grad()
    def timed_forward(self, batch):
        start_time = time()

        sources, hypotheses, features, scores = batch
        self.forward(sources, hypotheses, features)

        end_time = time()

        return end_time - start_time



    def configure_optimizers(self):

        return self.initialize_optimizer(self.parameters())
