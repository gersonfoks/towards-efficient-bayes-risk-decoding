from time import time

from torch.nn import Softplus

import pytorch_lightning as pl
import torch

from CustomLoss.GaussianMixtureLoss import GaussianMixtureLoss
from CustomLoss.StudentTMixtureLoss import StudentTMixtureLoss


class FullDecMixtureModel(pl.LightningModule):

    def __init__(self, full_dec_embedding, full_dec_pooling_layers,
                 token_statistics_pooling_layer, final_layers, initialize_optimizer, locs_activation_function, distribution='gaussian', device="cuda", n_components=2):
        super().__init__()
        self.device_name = device
        self.full_dec_embedding = full_dec_embedding
        self.full_dec_pooling_layers = full_dec_pooling_layers
        self.token_statistics_pooling_layer = token_statistics_pooling_layer

        self.final_layers = final_layers

        self.distribution = distribution
        if distribution == 'gaussian':
            self.criterion = GaussianMixtureLoss()
        elif distribution == 'student-t':
            print('using student-t distribution')
            self.criterion = StudentTMixtureLoss()
        else:
            raise ValueError('No distribution named: {}'.format(distribution))

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer
        self.name = 'full_dec_mixture_model'

        self.softplus = Softplus()
        self.locs_activation_function = locs_activation_function

        self.n_components = n_components

    def forward(self, sources, hypotheses, features):
        hidden_layer_embeddings, token_statistic_embedding, attention_mask = self.full_dec_embedding.forward(
            features["input_ids"],
            features["attention_mask"],
            features["decoder_input_ids"],
            features["labels"],
        )

        all_pooled_layers = [

        ]
        for emb, pooling_layer in zip(hidden_layer_embeddings, self.full_dec_pooling_layers):
            all_pooled_layers.append(pooling_layer(emb, attention_mask))

        pooled_statistics = self.token_statistics_pooling_layer(token_statistic_embedding, attention_mask)
        all_pooled_layers.append(pooled_statistics)

        final_features = torch.concat(all_pooled_layers, dim=-1)
        out = self.final_layers(final_features)

        if self.distribution == 'gaussian':
            locs = self.locs_activation_function(out[:, :self.n_components])
            scales = self.softplus(out[:, self.n_components:self.n_components * 2])

            logit_weights = out[:, 2 * self.n_components: 3 * self.n_components]

            return {
                'locs': locs,
                'scales': scales,
                'logit_weights': logit_weights,
            }
        elif self.distribution == 'student-t':
            locs = self.locs_activation_function(out[:, :self.n_components])
            scales = self.softplus(out[:, self.n_components:self.n_components * 2])

            logit_weights = out[:, 2 * self.n_components: 3 * self.n_components]

            degrees_of_freedom = self.softplus(out[:, 3 * self.n_components: 4 * self.n_components])

            return {
                'locs': locs,
                'degrees_of_freedom': degrees_of_freedom,
                'scales': scales,
                'logit_weights': logit_weights,
            }
        #print(torch.softmax(logit_weights, dim=-1))



    def batch_to_out(self, batch):

        sources, hypotheses, features, utilities = batch
        out = self.forward(sources, hypotheses, features)

        loss = 0
        if self.distribution == 'gaussian':
            loss = self.criterion(out['locs'], out['scales'], out['logit_weights'], utilities.to(self.device))
        elif self.distribution == 'student-t':
            loss = self.criterion(out['degrees_of_freedom'], out['locs'], out['scales'], out['logit_weights'], utilities.to(self.device))

        return {"loss": loss, **out}

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

