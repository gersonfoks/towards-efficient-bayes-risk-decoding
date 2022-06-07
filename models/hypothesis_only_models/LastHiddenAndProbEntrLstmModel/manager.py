import torch

from models.common.layers import EmbbedingForPackedSequenceLayer, get_feed_forward_layers, LastStateEmbedding
from models.common.optimization import get_optimizer_function
from models.hypothesis_only_models.HypothesisLstmModel.model import HypothesisLstmModel
from models.hypothesis_only_models.LastHiddenAndProbEntrLstmModel.model import LastHiddenAndProbEntrLstmModel
from models.hypothesis_only_models.LastHiddenLstmModel.LastHiddenLstmModel import LastHiddenLstmModel
from models.manager import ModelManager
from utilities.misc import load_nmt_model
from pathlib import Path

class LastHiddenAndProbEntrLstmModelManager(ModelManager):

    def __init__(self, config):
        super().__init__(config)
        self.config = config



    def create_model(self):
        config = self.config
        self.nmt_model, self.tokenizer = load_nmt_model(config["nmt_model"], pretrained=True)

        # Create the embedding layer

        embedding_size = 512

        embedding_layer = LastStateEmbedding(self.nmt_model)

        lstm_layer = torch.nn.LSTM(embedding_size, embedding_size, batch_first=True, bidirectional=True)
        prob_entropy_lstm_layer = torch.nn.LSTM(2, 128, bidirectional=True)

        final_layers = get_feed_forward_layers(config["feed_forward_layers"]["dims"],
                                               config["feed_forward_layers"]["activation_function"],
                                               config["feed_forward_layers"]["activation_function_last_layer"],
                                               config["dropout"],
                                               )

        initialize_optimizer = get_optimizer_function(config)
        self.model = LastHiddenAndProbEntrLstmModel(embedding_layer, lstm_layer, prob_entropy_lstm_layer, final_layers, initialize_optimizer)
        return self.model


