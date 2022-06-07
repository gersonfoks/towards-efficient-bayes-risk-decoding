import pytorch_lightning as pl
import torch
from torch.nn import MSELoss
from torch.nn.utils.rnn import pack_padded_sequence


class EncDecLastHiddenModel(pl.LightningModule):

    def __init__(self, last_hidden_state_embedding, enc_lstm_layer, dec_lstm_layer, final_layers, initialize_optimizer,
                 device="cuda", ):
        super().__init__()
        self.device_name = device

        self.last_hidden_state_embedding = last_hidden_state_embedding

        self.enc_lstm_layer = enc_lstm_layer
        self.dec_lstm_layer = dec_lstm_layer

        self.final_layers = final_layers

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

    def forward(self, sources, hypotheses, features):

        source_embeddings, attention_mask, hypotheses_embeddings, hypotheses_attention_mask = self.last_hidden_state_embedding.forward(
            features["input_ids"],
            features["attention_mask"],
            features["decoder_input_ids"],
            features["labels"], )

        ## We need to pack the embeddings
        source_lengths = features["source_sequence_lengths"].int().to("cpu")
        hypothesis_lengths = features["hypothesis_sequence_lengths"].int().to("cpu")

        source_packed_embeddings = pack_padded_sequence(source_embeddings, source_lengths, enforce_sorted=False,
                                                        batch_first=True)
        hypothesis_packed_embeddings = pack_padded_sequence(hypotheses_embeddings, hypothesis_lengths,
                                                            enforce_sorted=False, batch_first=True)

        _, (enc_h_n, _) = self.enc_lstm_layer(source_packed_embeddings)
        _, (dec_h_n, _) = self.dec_lstm_layer(hypothesis_packed_embeddings)

        enc_h_n = enc_h_n.permute(1, 0, 2).reshape(-1, 1024)
        dec_h_n = dec_h_n.permute(1, 0, 2).reshape(-1, 1024)

        features = torch.concat([enc_h_n, dec_h_n], dim=-1)

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
            self.log("train_{}".format(log_var), batch_out[log_var], batch_size=batch_size, on_step=True,
                     on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        batch_out = self.batch_to_out(batch)

        sources, hypotheses, features, scores = batch

        batch_size = len(scores)

        for log_var in self.log_vars:
            self.log("val_{}".format(log_var), batch_out[log_var], batch_size=batch_size, )

    def configure_optimizers(self):

        return self.initialize_optimizer(self.parameters())  # self.final_layers.parameters())

    def parameters(self, recursive=True):
        return iter(list(self.enc_lstm_layer.parameters()) + list(self.dec_lstm_layer.parameters()) + list(
            self.final_layers.parameters()))
