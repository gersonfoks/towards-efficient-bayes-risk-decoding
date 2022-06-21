import pytorch_lightning as pl
import torch
from torch.nn import MSELoss
from torch.nn.utils.rnn import pack_padded_sequence


class FullDecCometModel(pl.LightningModule):

    def __init__(self, hidden_state_embedding, hidden_state_lstms, prob_entropy_lstm_layer, comet_linear_layers,
                 final_layers, initialize_optimizer, device="cuda", ):
        super().__init__()
        self.device_name = device

        self.hidden_state_embedding = hidden_state_embedding

        self.hidden_state_lstms = hidden_state_lstms

        # Register the parameters
        parameter_list = []
        for lstm in self.hidden_state_lstms:
            parameter_list += list(lstm.parameters())

        self.registered_lstms = torch.nn.ParameterList(parameter_list)

        self.prob_entropy_lstm_layer = prob_entropy_lstm_layer

        self.comet_linear_layers = comet_linear_layers

        self.final_layers = final_layers

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

    def forward_comet_features(self, features):
        h_emb = features["hypothesis_embedding"]
        s_emb = features["source_embedding"]

        comet_features = [
            h_emb,
            s_emb,
            torch.abs(s_emb - h_emb),
            h_emb * s_emb,

        ]

        for r_emb in features["reference_embeddings"]:
            h_r_prod = h_emb * r_emb
            h_r_diff = torch.abs(h_emb - r_emb)

            comet_features += [r_emb, h_r_prod, h_r_diff]

        comet_features = torch.concat(comet_features, dim=-1)

        return self.comet_linear_layers(comet_features)

    def forward_through_dec_model(self, features):
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

        return features

    def forward(self, sources, hypotheses, features):

        # The features for
        dec_features = self.forward_through_dec_model(features)
        # The features for comet
        comet_features = self.forward_comet_features(features)

        final_features = torch.concat([dec_features, comet_features], dim=-1)

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
