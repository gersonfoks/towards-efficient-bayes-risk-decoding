from typing import Iterator

import torch
from torch import nn

activation_functions = {
    'silu': nn.SiLU,
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid
}


def get_feed_forward_layers(layer_dims, activation_function, activation_function_last_layer=None, dropout=0.0):
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

    if activation_function_last_layer != None:
        activation_function_last_layer = activation_functions[activation_function_last_layer]
        layers.append(activation_function_last_layer())
    return nn.Sequential(*layers)




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


    def forward(self, input_ids=None, attention_mask=None,  decoder_input_ids=None, labels=None):

        with torch.no_grad():
            nmt_out = self.nmt_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                             decoder_input_ids=decoder_input_ids, output_hidden_states=True,
                                             output_attentions=True)
        attention_mask_decoder = (self.padding_id != labels).long()
        return nmt_out["decoder_hidden_states"][-1], attention_mask_decoder

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
        return  nmt_out["encoder_hidden_states"][-1], attention_mask, nmt_out["decoder_hidden_states"][-1], attention_mask_decoder,



class HiddenStateEmbedding(nn.Module):

    def __init__(self, nmt_model, padding_id=58100, n_layers = 7):
        super().__init__()
        self.nmt_model = nmt_model
        self.padding_id = padding_id

        weights = torch.empty(1, n_layers)
        nn.init.normal_(weights)




        self.weights = nn.Parameter(weights.unsqueeze(dim=-1) , requires_grad=True)
        self.softmax = torch.nn.Softmax(dim=-1)



    def take_avg(self, hidden_state, attention_mask):
        hidden_state_sum = torch.sum(hidden_state * attention_mask.unsqueeze(dim=-1), dim=1)

        return hidden_state_sum / torch.sum(attention_mask)

    def forward(self, input_ids=None, attention_mask=None,  decoder_input_ids=None, labels=None):

        with torch.no_grad():
            nmt_out = self.nmt_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                             decoder_input_ids=decoder_input_ids, output_hidden_states=True,
                                             output_attentions=True)
        attention_mask_decoder = (self.padding_id != labels).long()

        #Calculate the average hidden state


        # First we take the avg of each hidden state:

        hidden_states = [self.take_avg(h, attention_mask_decoder) for h in nmt_out["decoder_hidden_states"]]


        hidden_states = torch.stack(hidden_states, dim=1) * self.softmax(self.weights)

        hidden_states = torch.sum(hidden_states,dim=1) # Get the avg

        return hidden_states

    def parameters(self, recurse: bool = True):
        return iter([self.weights])