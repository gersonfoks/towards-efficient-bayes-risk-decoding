import pytorch_lightning as pl
import torch
from torch.nn import MSELoss



class PropEntropyLstmModel(pl.LightningModule):

    def __init__(self,  prob_lstm_layer, entropy_lstm_layer, final_layers, initialize_optimizer, device="cuda", ):
        super().__init__()
        self.device_name = device



        self.prob_lstm_layer = prob_lstm_layer
        self.entropy_lstm_layer = entropy_lstm_layer

        self.final_layers = final_layers

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

    def forward(self, sources, hypotheses, features):

        _, (probs_h_n, _) = self.prob_lstm_layer(features["log_prob"])
        _, (entropy_h_n, _) = self.entropy_lstm_layer(features["entropy"])


        probs_h_n = probs_h_n.permute(1,0, 2).reshape(-1, 256)
        entropy_h_n = entropy_h_n.permute(1,0, 2).reshape(-1, 256)


        features = torch.concat([probs_h_n, entropy_h_n], dim=-1)


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
            self.log("train_{}".format(log_var), batch_out[log_var], batch_size=batch_size, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        batch_out = self.batch_to_out(batch)

        sources, hypotheses, features, scores = batch

        batch_size = len(scores)

        for log_var in self.log_vars:
            self.log("val_{}".format(log_var), batch_out[log_var], batch_size=batch_size,)

    @torch.no_grad()
    def predict(self, sources, hypotheses, references):
        '''
        Predicts the bayes risk for the source targets pairs
        :param sources:
        :param targets:
        :return:
        '''

        pass

    def configure_optimizers(self):

        return self.initialize_optimizer(self.parameters())  # self.final_layers.parameters())

    def parameters(self, recursive=True):
        return iter(list(self.prob_lstm_layer.parameters()) + list(self.entropy_lstm_layer.parameters()) + list(self.final_layers.parameters()))
