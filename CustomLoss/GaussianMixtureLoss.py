# Inspiration from: https://colab.research.google.com/drive/19-APMZs6LX-iQnjQfEpFXbjMSWs5c4ub?usp=sharing


from torch import nn
import torch.distributions as td


class GaussianMixtureLoss(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, loc, scale, weights, utilities):


        components = self.make_components(loc, scale, utilities.shape[-1])

        mixture = td.MixtureSameFamily(td.Categorical(logits=weights), components)

        log_p = mixture.log_prob(utilities)
        loss = -log_p.mean(0)
        # Divide by the number of independent samples to make the interpretation of the results easier
        return loss / utilities.shape[-1]

    def make_components(self, loc, scale, sample_size):
        shape = loc.shape
        loc = loc.unsqueeze(-1).repeat((1,) * len(shape) + (sample_size,))
        scale = scale.unsqueeze(-1).repeat((1,) * len(shape) + (sample_size,))
        return td.Independent(td.Normal(loc=loc, scale=scale), 1)