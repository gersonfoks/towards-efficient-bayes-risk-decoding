import pytorch_lightning as pl
import torch
from torch.nn import MSELoss
from torch.nn.utils.rnn import pack_padded_sequence


class LastHiddenStateRefModel(pl.LightningModule):

    def __init__(self, last_hidden_state_embedding, lstm_hypothesis, lstm_references, final_layers, initialize_optimizer, device="cuda", ):
        super().__init__()
        self.device_name = device

        self.last_hidden_state_embedding = last_hidden_state_embedding

        self.lstm_hypothesis = lstm_hypothesis
        self.lstm_references = lstm_references

        self.final_layers = final_layers

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

    def forward(self, sources, hypotheses, features):
        # Embed the hypothesis
        embedded_hypotheses, attention_mask = self.last_hidden_state_embedding.forward(features["input_ids"],
                                                                              features["attention_mask"],
                                                                              features["decoder_input_ids"],
                                                                              features["labels"],

                                                                              )

        lengths = features["hypotheses_sequence_lengths"].int().to("cpu")

        hypotheses_packed = pack_padded_sequence(embedded_hypotheses, lengths, enforce_sorted=False, batch_first=True)

        # Map the references to embedding space
        embedded_references = []
        for references, seq_lengths in zip(features["reference_inputs"], features["ref_sequence_lengths"]):
            embedded_reference, attention_mask = self.last_hidden_state_embedding.forward(references["input_ids"],
                                                                                           references["attention_mask"],
                                                                                           references[
                                                                                               "decoder_input_ids"],
                                                                                           references["labels"],
                                                                                           )
            seq_lengths = seq_lengths.int().to("cpu")
            packed_embedded_refs = pack_padded_sequence(embedded_reference, seq_lengths, enforce_sorted=False, batch_first=True)

            embedded_references.append(packed_embedded_refs)


        # Next get the last hidden states
        _, (h_n_hypothesis, _) = self.lstm_hypothesis(hypotheses_packed)
        h_n_hypothesis = h_n_hypothesis.permute(1, 0, 2).reshape(-1, 512)
        # Also for the references

        reference_hidden_states = []
        for embedded_reference in embedded_references:

            _, (h_n_embedding, _) = self.lstm_references(embedded_reference)

            h_n_embedding = h_n_embedding.permute(1, 0, 2).reshape(-1, 512)

            reference_hidden_states.append(h_n_embedding)

        all_hidden_states = [h_n_hypothesis] + reference_hidden_states
        # Lastly we need to concatonate everything
        features = torch.concat(all_hidden_states, dim=-1)




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
