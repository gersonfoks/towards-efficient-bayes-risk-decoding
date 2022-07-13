import torch

from models.common.layers import get_feed_forward_layers, LastStateEmbedding
from models.common.optimization import get_optimizer_function

from models.Base.BaseManager import BaseManager
from models.old.reference_models.LastHiddenStateRefModel.model import LastHiddenStateRefModel

from utilities.misc import load_nmt_model


class LastHiddenStateRefBaseManager(BaseManager):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def create_model(self):
        config = self.config
        self.nmt_model, self.tokenizer = load_nmt_model(config["nmt_model"], pretrained=True)

        # Create the embedding layer

        embedding_size = 512

        embedding_layer = LastStateEmbedding(self.nmt_model)

        lstm_hypothesis = torch.nn.LSTM(embedding_size, 256, bidirectional=True)
        lstm_references_hypothesis = torch.nn.LSTM(embedding_size, 256, bidirectional=True)

        final_layers = get_feed_forward_layers(config["feed_forward_layers"]["dims"],
                                               config["feed_forward_layers"]["activation_function"],
                                               config["feed_forward_layers"]["activation_function_last_layer"],
                                               config["dropout"],
                                               )

        initialize_optimizer = get_optimizer_function(config)
        self.model = LastHiddenStateRefModel(embedding_layer, lstm_hypothesis, lstm_references_hypothesis, final_layers, initialize_optimizer)
        return self.model
