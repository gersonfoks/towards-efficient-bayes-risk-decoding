import pytorch_lightning as pl
import torch
from torch.nn import MSELoss
from torch.nn.utils.rnn import pack_padded_sequence


class FullDecHiddenLstmModel(pl.LightningModule):

    def __init__(self, hidden_state_embedding, hidden_state_lstms, prob_entropy_lstm_layer, final_layers, initialize_optimizer,
                 device="cuda", ):
        super().__init__()
        self.device_name = device

        self.hidden_state_embedding = hidden_state_embedding

        self.hidden_state_lstms = hidden_state_lstms

        self.prob_entropy_lstm_layer = prob_entropy_lstm_layer

        self.final_layers = final_layers

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

    def forward(self, sources, hypotheses, features):
        embeddings, _ = self.hidden_state_embedding.forward(features["input_ids"],
                                                                         features["attention_mask"],
                                                                         features["decoder_input_ids"],
                                                                         features["labels"],
                                                                         )
        ## We need to pack the embeddings
        lengths = features["sequence_lengths"].int().to("cpu")

        hidden_states = []

        for embedding, lstm in zip(embeddings, self.hidden_state_lstms):
            packed_embeddings = pack_padded_sequence(embedding, lengths, enforce_sorted=False, batch_first=True)

            _, (l_h_n, _) = lstm(packed_embeddings)
            l_h_n = l_h_n.permute(1, 0, 2).reshape(-1, 256)
            hidden_states.append(l_h_n)

        _, (probs_entropy_h_n, _) = self.prob_entropy_lstm_layer(features["log_prob_entropy"])

        probs_entropy_h_n = probs_entropy_h_n.permute(1, 0, 2).reshape(-1, 256)

        hidden_states.append(probs_entropy_h_n)
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
        lstm_layer_list = []
        for lstm in self.hidden_state_lstms:
            lstm_layer_list += list(lstm.parameters())
        return iter(lstm_layer_list + list(self.prob_entropy_lstm_layer.parameters()) + list(self.final_layers.parameters()))
