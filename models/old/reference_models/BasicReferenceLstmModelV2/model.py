import pytorch_lightning as pl
import torch
from torch.nn import MSELoss


class BasicReferenceLstmModelV2(pl.LightningModule):

    def __init__(self, embedding_layer, lstm_hypothesis, lstm_references, prob_entropy_layer,  final_layers, initialize_optimizer, device="cuda", ):
        super().__init__()
        self.device_name = device

        self.embedding_layer = embedding_layer

        self.prob_entropy_layer = prob_entropy_layer
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
        embedded_hypotheses = self.embedding_layer(features["tokenized_hypotheses"])

        # Map the references to embedding space
        embedded_references = []
        for references in features["tokenized_references"]:
            embedded_references.append(self.embedding_layer(references))

        #Next we concatonate all the

        # Next get the last hidden states
        _, (h_n_hypothesis, _) = self.lstm_hypothesis(embedded_hypotheses)
        h_n_hypothesis = h_n_hypothesis.permute(1, 0, 2).reshape(-1, 512)
        # Also for the references

        reference_hidden_states = []
        for embedded_reference in embedded_references:

            _, (h_n_embedding, _) = self.lstm_references(embedded_reference)

            h_n_embedding = h_n_embedding.permute(1, 0, 2).reshape(-1, 512)

            reference_hidden_states.append(h_n_embedding)

        all_hidden_states = [h_n_hypothesis] + reference_hidden_states


        # Furthermore we add the probabilities and entropy of each sentence:

        prob_and_entropies = features["probs_and_entropy"]
        # Put it through a linear layer
        prob_and_entropies = self.prob_entropy_layer(prob_and_entropies)

        # Lastly we need to concatonate everything
        features = torch.concat(all_hidden_states + [prob_and_entropies], dim=-1)


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
