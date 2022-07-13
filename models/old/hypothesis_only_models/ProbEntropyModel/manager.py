import torch

from models.common.layers import get_feed_forward_layers
from models.common.optimization import get_optimizer_function
from models.old.hypothesis_only_models.ProbEntropyModel.model import ProbEntropyModel
from models.Base.BaseManager import BaseManager
from utilities.misc import load_nmt_model
from pathlib import Path

class ProbEntropyBaseManager(BaseManager):

    def __init__(self, config):
        super().__init__(config)
        self.config = config



    def create_model(self):
        config = self.config
        self.nmt_model, self.tokenizer = load_nmt_model(config["nmt_model"], pretrained=True)

        # Create the embedding layer

        embedding_size = 1

        hidden_dim = config["lstms"]["hidden_dim"]

        probability_lstm_layer = torch.nn.LSTM(embedding_size, hidden_dim, bidirectional=True)
        entropy_lstm_layer = torch.nn.LSTM(embedding_size, hidden_dim, bidirectional=True)


        final_layers = get_feed_forward_layers(config["feed_forward_layers"]["dims"],
                                               config["feed_forward_layers"]["activation_function"],
                                               config["feed_forward_layers"]["activation_function_last_layer"],
                                               config["dropout"],
                                               )

        initialize_optimizer = get_optimizer_function(config)
        self.model = ProbEntropyModel(probability_lstm_layer, entropy_lstm_layer, final_layers, initialize_optimizer)
        return self.model


    def save_model(self, save_model_path):
        Path(save_model_path).mkdir(parents=True, exist_ok=True)
        pl_path = save_model_path + 'pl_model.pt'

        state = {
            "config": self.config,
            "state_dict": self.model.state_dict()
        }

        torch.save(state, pl_path)

    @classmethod
    def load_model(cls, model_path):
        '''
        Returns a manager and a loaded model specified by the file pointed by the model_path
        :param model_path:
        :return:
        '''

        pl_path = model_path + 'pl_model.pt'
        checkpoint = torch.load(pl_path)
        manager = ProbEntropyBaseManager(checkpoint["config"])
        model = manager.create_model()

        model.load_state_dict(checkpoint["state_dict"])
        return model, manager