from typing import Iterator

import torch
from torch import nn

activation_functions = {
    'silu': nn.SiLU,
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,

}


def get_feed_forward_layers(layer_dims, activation_function, activation_function_last_layer=None, dropout=0.0, batch_norm=False, last_layer_scale=None):
    '''
    Creates feed forward layers with dimensions defined in layer_dims.
    :return: 
    '''

    activation_function = activation_functions[activation_function]

    layers = []
    # Add all the layers except the last one
    if batch_norm:
        print("using_batchnorm for the first layer")
        layers.append(nn.BatchNorm1d(layer_dims[0]))
    for layer_in, layer_out in zip(layer_dims[:-2], layer_dims[1:-1]):
        layers.append(nn.Linear(layer_in, layer_out))
        if batch_norm:
            print("using_batchnorm for the first layer")
            layers.append(nn.BatchNorm1d(layer_out))
        layers.append(activation_function())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    # Add the last one
    layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))

    if activation_function_last_layer != None:


        activation_function= activation_functions[activation_function_last_layer]


        # If we need to scale (used for comet models)
        if last_layer_scale != None:
            layers.append(
                ScaledActivationFucntion(activation_function(), last_layer_scale)
            )
        else:
            layers.append(activation_function())
    return nn.Sequential(*layers)


class ScaledActivationFucntion(nn.Module):

    def __init__(self, f, scale):
        super().__init__()
        self.f = f
        self.scale = scale

    def forward(self, x):
        return self.f(x) * self.scale




class EmbbedingForPackedSequenceLayer(torch.nn.Module):
    """
    Lstm embedding layer that handles packed sequences
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding = torch.nn.Embedding(input_dim, output_dim)

    def forward(self, packed_sequence):
        return self.element_wise_apply(self.embedding, packed_sequence)

    def element_wise_apply(self, fn, packed_sequence):
        # from: https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-are-going-to-use-word-embedding-and-bilstm/28184/3
        """applies a pointwise function fn to each element in packed_sequence"""

        return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes,
                                                 sorted_indices=packed_sequence.sorted_indices,
                                                 unsorted_indices=packed_sequence.unsorted_indices)


class LastStateEmbedding(nn.Module):

    def __init__(self, nmt_model, padding_id=58100):
        super().__init__()
        self.nmt_model = nmt_model
        self.padding_id = padding_id

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None):
        with torch.no_grad():
            nmt_out = self.nmt_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                             decoder_input_ids=decoder_input_ids, output_hidden_states=True,
                                             output_attentions=True)
        attention_mask_decoder = (self.padding_id != labels).long()
        return nmt_out["decoder_hidden_states"][-1], attention_mask_decoder


class HiddenStateEmbedding(nn.Module):

    def __init__(self, nmt_model, padding_id=58100):
        super().__init__()
        self.nmt_model = nmt_model
        self.padding_id = padding_id

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None):
        with torch.no_grad():
            nmt_out = self.nmt_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                             decoder_input_ids=decoder_input_ids, output_hidden_states=True,
                                             output_attentions=True)
        attention_mask_decoder = (self.padding_id != labels).long()
        # print(list(nmt_out["decoder_hidden_states"]))
        return nmt_out["decoder_hidden_states"], attention_mask_decoder


class EncDecLastStateEmbedding(nn.Module):

    def __init__(self, nmt_model, padding_id=58100):
        super().__init__()
        self.nmt_model = nmt_model
        self.padding_id = padding_id

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None):
        with torch.no_grad():
            nmt_out = self.nmt_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                             decoder_input_ids=decoder_input_ids, output_hidden_states=True,
                                             output_attentions=True)
        attention_mask_decoder = (self.padding_id != labels).long()
        return nmt_out["encoder_hidden_states"][-1], attention_mask, nmt_out["decoder_hidden_states"][
            -1], attention_mask_decoder,


class GlobalMaxPooling(nn.Module):

    def forward(self, x, padding=None):

        if padding != None:

            x = x + x * (~ padding * (- 1e6)).unsqueeze(-1)# Very big negative number

        out = torch.max(x, dim=1).values

        return out


class GlobalMeanPooling(nn.Module):

    def forward(self, x, padding=None):
        padding = ~padding.unsqueeze(-1)
        if padding != None:

            x = x + (x * padding)
        x_summed = torch.sum(x, dim=1)
        normalizing_constant = torch.sum(padding.squeeze(-1), dim=-1).unsqueeze(-1)


        out =  x_summed / normalizing_constant

        return out


class WeightedBagEmbeddingSequence(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)


    def forward(self, indices, weights):

        '''
        Long tensor containing indices of the tokens with size = B * s * n
        Weights are the weights for each tensor with size = B * s * n
        '''

        shape = indices.shape
        new_shape = (shape[0], -1)
        indices_reshaped = indices.reshape(new_shape)

        embeddings = self.embedding(indices_reshaped)

        # Reformat to the right shape again
        embeddings = embeddings.reshape(shape[0], shape[1], self.embedding_dim)

        # Take the weighted average of the last dimension.
        weighted_embedding = torch.sum(embeddings * weights, dim=-1)

        return weighted_embedding



class WeightedBagEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)


    def forward(self, indices, weights):

        '''
        Long tensor containing indices of the tokens with size = B * n
        Weights are the weights for each tensor with size = B * n
        '''

        embeddings = self.embedding(indices)


        # Take the weighted average of the last dimension.
        weighted_embedding = torch.sum(embeddings * weights, dim=-1)

        return weighted_embedding



class LearnedPoolingLayer(nn.Module):

    def __init__(self, embedding_size, n_heads=4):

        super().__init__()
        self.embedding_size = embedding_size
        query_tensor = torch.zeros(1, 1, embedding_size)
        nn.init.xavier_normal_(query_tensor)
        self.query = torch.nn.Parameter(query_tensor)
        self.attention = torch.nn.MultiheadAttention(embedding_size, n_heads, batch_first=True)

    def forward(self, x, att_mask):
        query = self.query.repeat(x.shape[0], 1, 1)

        hidden_state, _ = self.attention(query=query, key=x,
                                                     value=x,
                                                     key_padding_mask=~att_mask.bool(),
                                                     )
        return hidden_state


def get_learnable_embeddig(shape):
    emb = torch.zeros(shape)  # Use two embeddings instead of one
    nn.init.xavier_normal_(emb)
    learnable_embedding = torch.nn.Parameter(emb)
    return learnable_embedding

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

        query_attention = torch.concat([torch.ones(batch_size, self.n_learnable_embeddings).to("cuda"), query_attention], dim=-1)
        key_attention = torch.concat([torch.ones(batch_size, self.n_learnable_embeddings).to("cuda"), key_attention], dim=-1)

        att_out, _ = self.attention(query=query, key=key, value=value, key_padding_mask=~key_attention.bool())

        # pool
        pooled_out = self.pooling(att_out,  query_attention.bool())
        return pooled_out








