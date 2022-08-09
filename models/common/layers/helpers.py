from typing import Iterator

import torch
from torch import nn


from models.common.layers.pooling import LearnedPoolingLayer

activation_functions = {
    'silu': nn.SiLU,
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,

}


def get_feed_forward_layers(layer_dims, activation_function, activation_function_last_layer=None, dropout=0.0,
                            last_layer_scale=None):
    '''
    Creates feed forward layers with dimensions defined in layer_dims.
    :return: 
    '''

    activation_function = activation_functions[activation_function]

    layers = []
    # Add all the layers except the last one

    for layer_in, layer_out in zip(layer_dims[:-2], layer_dims[1:-1]):
        layers.append(nn.Linear(layer_in, layer_out))
        layers.append(activation_function())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    # Add the last one
    layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))

    if activation_function_last_layer != None and activation_function_last_layer != 'none':

        activation_function = activation_functions[activation_function_last_layer]

        # If we need to scale (used for comet models)
        if last_layer_scale != None:
            print("get scaled activation funcion")
            layers.append(
                ScaledActivationFunction(activation_function(), last_layer_scale)
            )
        else:
            layers.append(activation_function())
    return nn.Sequential(*layers)


class ScaledActivationFunction(nn.Module):

    def __init__(self, f, scale):
        super().__init__()
        self.f = f
        self.scale = scale

    def forward(self, x):
        return self.f(x) * self.scale


class AttentionWithLearnableEmbedding(nn.Module):

    def __init__(self, query_emb_size, key_emb_size, value_emb_size, n_learnable_embeddings, num_heads=4):
        super().__init__()

        self.n_learnable_embeddings = n_learnable_embeddings

        self.learnable_embedding_query = get_learnable_embeddig((1, n_learnable_embeddings, query_emb_size))
        self.learnable_embedding_key = get_learnable_embeddig((1, n_learnable_embeddings, key_emb_size))
        self.learnable_embedding_value = get_learnable_embeddig((1, n_learnable_embeddings, value_emb_size))

        self.attention = nn.MultiheadAttention(query_emb_size, num_heads, batch_first=True, vdim=value_emb_size)

        self.pooling = LearnedPoolingLayer(query_emb_size, )

    def forward(self, query, key, value, query_attention, key_attention):
        batch_size = query.shape[0]
        learnable_emb_query = self.learnable_embedding_query.repeat(batch_size, 1, 1)
        learnable_embedding_key = self.learnable_embedding_key.repeat(batch_size, 1, 1)
        learnable_embedding_value = self.learnable_embedding_value.repeat(batch_size, 1, 1)

        query = torch.concat([learnable_emb_query, query], dim=1)
        key = torch.concat([learnable_embedding_key, key], dim=1)
        value = torch.concat([learnable_embedding_value, value], dim=1)

        query_attention = torch.concat(
            [torch.ones(batch_size, self.n_learnable_embeddings).to("cuda"), query_attention], dim=-1)
        key_attention = torch.concat([torch.ones(batch_size, self.n_learnable_embeddings).to("cuda"), key_attention],
                                     dim=-1)

        att_out, _ = self.attention(query=query, key=key, value=value, key_padding_mask=~key_attention.bool())

        # pool
        pooled_out = self.pooling(att_out, query_attention.bool())
        return pooled_out







def get_learnable_embeddig(shape):
    emb = torch.zeros(shape)
    nn.init.xavier_normal_(emb)
    learnable_embedding = torch.nn.Parameter(emb)
    return learnable_embedding
