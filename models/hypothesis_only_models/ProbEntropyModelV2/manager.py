import torch

from models.ProbEntropyModelV2.model import ProbEntropyModelV2
from models.common.layers import EmbbedingForPackedSequenceLayer, get_feed_forward_layers, LastStateEmbedding
from models.common.optimization import get_optimizer_function
from models.hypothesis_only_models.HypothesisLstmModel.model import HypothesisLstmModel
from models.hypothesis_only_models.LastHiddenLstmModel.LastHiddenLstmModel import LastHiddenLstmModel
from models.hypothesis_only_models.ProbEntropyModel.model import ProbEntropyModel
from models.manager import ModelManager
from utilities.misc import load_nmt_model
from pathlib import Path

class ProbEntropyModelManagerV2(ModelManager):

    def __init__(self, config):
        super().__init__(config)
        self.config = config



    def create_model(self):
        config = self.config
        self.nmt_model, self.tokenizer = load_nmt_model(config["nmt_model"], pretrained=True)

        # Create the embedding layer

        embedding_size = 2

        hidden_dim = config["lstms"]["hidden_dim"]

        lstm_layer = torch.nn.LSTM(embedding_size, hidden_dim, bidirectional=True)



        final_layers = get_feed_forward_layers(config["feed_forward_layers"]["dims"],
                                               config["feed_forward_layers"]["activation_function"],
                                               config["feed_forward_layers"]["activation_function_last_layer"],
                                               config["dropout"],
                                               )

        initialize_optimizer = get_optimizer_function(config)
        self.model = ProbEntropyModelV2(lstm_layer, final_layers, initialize_optimizer)
        return self.model


    def save_model(self, save_model_path):
        Path(save_model_path).mkdir(parents=True, exist_ok=True)
        pl_path = save_model_path + 'pl_model.pt'

        state = {
            "config": self.config,
            "state_dict": self.model.state_dict()
        }

        torch.save(state, pl_path)

