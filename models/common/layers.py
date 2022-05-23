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
    Lstm embedding layer that handels packed sequences
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
        # embeddings = fn(packed_sequence.data)
        # print(embeddings)
        # print(embeddings.shape)
        return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes,
                                                 sorted_indices=packed_sequence.sorted_indices,
                                                 unsorted_indices=packed_sequence.unsorted_indices)
