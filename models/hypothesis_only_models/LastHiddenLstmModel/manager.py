import torch

from models.common.layers import EmbbedingForPackedSequenceLayer, get_feed_forward_layers, LastStateEmbedding
from models.common.optimization import get_optimizer_function
from models.hypothesis_only_models.HypothesisLstmModel.model import HypothesisLstmModel
from models.hypothesis_only_models.LastHiddenLstmModel.LastHiddenLstmModel import LastHiddenLstmModel
from models.manager import ModelManager
from utilities.misc import load_nmt_model
from pathlib import Path

class LastHiddenLstmManager(ModelManager):

    def __init__(self, config):
        self.config = config



    def create_model(self):
        config = self.config
        self.nmt_model, self.tokenizer = load_nmt_model(config["nmt_model"], pretrained=True)

        # Create the embedding layer

        embedding_size = 512

        embedding_layer = LastStateEmbedding(self.nmt_model)

        lstm_layer = torch.nn.LSTM(embedding_size, embedding_size, batch_first=True, bidirectional=True)

        final_layers = get_feed_forward_layers(config["feed_forward_layers"]["dims"],
                                               config["feed_forward_layers"]["activation_function"],
                                               config["feed_forward_layers"]["activation_function_last_layer"],
                                               config["dropout"],
                                               )

        initialize_optimizer = get_optimizer_function(config)
        self.model = LastHiddenLstmModel(embedding_layer, lstm_layer, final_layers, initialize_optimizer)
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
        manager = LastHiddenLstmManager(checkpoint["config"])
        model = manager.create_model()

        model.load_state_dict(checkpoint["state_dict"])
        return model, manager