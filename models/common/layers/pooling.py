import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class LstmPoolingLayer(nn.Module):

    def __init__(self, embedding_size, hidden_state_size):
        super().__init__()
        self.hidden_state_size = hidden_state_size
        self.lstm_layer = nn.LSTM(embedding_size, hidden_state_size, bidirectional=True, batch_first=True)

    def forward(self, x, attention):
        lengths = attention.sum(-1).long().cpu()  # Get the length of each thing in x

        x_packed = pack_padded_sequence(x, lengths, enforce_sorted=False, batch_first=True)

        _, (h_n, _) = self.lstm_layer(x_packed)

        # Create feature vector out of it
        return h_n.permute(1, 0, 2).reshape(-1, self.hidden_state_size * 2)


class LearnedPoolingLayer(nn.Module):

    def __init__(self, embedding_size, n_queries=1, n_heads=4):
        super().__init__()
        self.embedding_size = embedding_size
        self.n_queries = n_queries
        query_tensor = torch.zeros(1, n_queries, embedding_size)
        nn.init.xavier_normal_(query_tensor)
        self.query = torch.nn.Parameter(query_tensor)
        self.attention = torch.nn.MultiheadAttention(embedding_size, n_heads, batch_first=True)

    def forward(self, x, att_mask):
        query = self.query.repeat(x.shape[0], 1, 1)
        hidden_state, _ = self.attention(query=query, key=x,
                                         value=x,
                                         key_padding_mask=~att_mask.bool(),
                                         )

        if self.n_queries > 1:
            hidden_state = hidden_state.reshape(-1, self.embedding_size * self.n_queries)
        return hidden_state


class GlobalMaxPooling(nn.Module):

    def forward(self, x, padding=None):

        if padding != None:
            x = x + x * (~ padding * (- 1e6)).unsqueeze(-1)  # Very big negative number

        out = torch.max(x, dim=1).values

        return out


class GlobalMeanPooling(nn.Module):

    def forward(self, x, padding=None):
        '''
        Gets the mean value of the last layer
        :param x: vectors to pool
        :param padding: 1 if there is padding 0 otherwise
        :return:
        '''
        padding = ~padding.unsqueeze(-1)

        if padding != None:

            x = x * padding
        x_summed = torch.sum(x, dim=1)
        normalizing_constant = torch.sum(padding.squeeze(-1), dim=-1).unsqueeze(-1)

        out = x_summed / normalizing_constant

        return out


class FullDecPooling(nn.Module):

    def __init__(self, full_dec_embedding, token_statistics_embedding, full_dec_pooling_layers,
                 token_statistics_pooling_layer):
        super().__init__()
        self.token_statistics_embedding = token_statistics_embedding
        self.full_dec_embedding = full_dec_embedding
        self.full_dec_pooling_layers = full_dec_pooling_layers
        self.token_statistics_pooling_layer = token_statistics_pooling_layer

    def forward(self, features):
        embeddings, attention_mask = self.full_dec_embedding.forward(features["input_ids"],
                                                                     features["attention_mask"],
                                                                     features["decoder_input_ids"],
                                                                     features["labels"],
                                                                     )

        all_pooled_layers = [

        ]
        for emb, pooling_layer in zip(embeddings, self.full_dec_pooling_layers):
            all_pooled_layers.append(pooling_layer(emb, attention_mask))

        statistics_embeddings = self.token_statistics_embedding.forward(features["token_statistics"], )

        pooled_statistics = self.token_statistics_pooling_layer(statistics_embeddings,
                                                                features["attention_mask_tokens"])
        all_pooled_layers.append(pooled_statistics)

        nmt_features = torch.concat(all_pooled_layers, dim=-1)

        return nmt_features
