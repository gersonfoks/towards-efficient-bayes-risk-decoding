import torch
from torch.nn import MSELoss
from torch.nn.utils.rnn import pack_padded_sequence

from models.Base.BaseModel import BaseModel


class FullDecUtilityBaseModel(BaseModel):

    def __init__(self, hidden_state_embedding, hidden_state_lstms, prob_entropy_lstm_layer, final_layers, utility_fn, initialize_optimizer,
                 device="cuda", ):
        super().__init__()
        self.device_name = device

        self.hidden_state_embedding = hidden_state_embedding

        self.hidden_state_lstms = hidden_state_lstms

        self.utility = utility_fn

        # Register the parameters
        parameter_list = []
        for lstm in self.hidden_state_lstms:
            parameter_list += list(lstm.parameters())

        self.registered_lstms = torch.nn.ParameterList(parameter_list)

        self.prob_entropy_lstm_layer = prob_entropy_lstm_layer

        self.final_layers = final_layers

        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

    def forward(self, sources, hypotheses, features):
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


        #Next we add the utility outcome


        #scores = torch.tensor(self.utility.call_batched(sources, hypotheses, features["references"])).to("cuda").squeeze(dim=1)

        # We already have the scores
        scores = features["utilities"]


        features = torch.concat(hidden_states + [scores], dim=-1)

        predicted_scores = self.final_layers(features)

        return predicted_scores

