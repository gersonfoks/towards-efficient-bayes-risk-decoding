import pytorch_lightning as pl
import torch
from torch.nn import MSELoss
from torch.nn.utils.rnn import pack_padded_sequence


class LastHiddenLstmModel(pl.LightningModule):

    def __init__(self, last_hidden_state_embedding, lstm_layer, final_layers, initialize_optimizer, device="cuda", ):
        super().__init__()
        self.device_name = device

        self.last_hidden_state_embedding = last_hidden_state_embedding

        self.lstm_layer = lstm_layer

        self.final_layers = final_layers

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

    def forward(self, sources, hypotheses, features):

        embeddings, attention_mask = self.last_hidden_state_embedding.forward(features["input_ids"],
                                                                              features["attention_mask"],
                                                                              features["decoder_input_ids"],
                                                                              features["labels"],

                                                                              )

        ## We need to pack the embeddings
        lengths = features["sequence_lengths"].int().to("cpu")

        packed_embeddings = pack_padded_sequence(embeddings, lengths, enforce_sorted=False, batch_first=True)

        output, (h_n, c_n) = self.lstm_layer(packed_embeddings)

        h_n = h_n.permute(1,0, 2).reshape(-1, 1024)

        predicted_scores = self.final_layers(h_n)

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
        return iter(list(self.lstm_layer.parameters()) + list(self.final_layers.parameters()))
