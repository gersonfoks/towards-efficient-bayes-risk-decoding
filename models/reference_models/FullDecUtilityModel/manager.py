import torch

from models.common.layers import get_feed_forward_layers, HiddenStateEmbedding
from models.common.optimization import get_optimizer_function

from models.Base.BaseManager import BaseManager
from models.reference_models.FullDecUtilityModel.model import FullDecUtilityBaseModel
from scripts.preprocessing.create_bayes_risk_dataset import load_utility
from utilities.misc import load_nmt_model


class FullDecUtilityBaseManager(BaseManager):

    def __init__(self, config):
        super().__init__(config)
        self.config = config



    def create_model(self):
        config = self.config
        self.nmt_model, self.tokenizer = load_nmt_model(config["nmt_model"], pretrained=True)

        utility = load_utility("unigram-f1", tokenizer=self.tokenizer)

        # Create the embedding layer

        embedding_size = 128

        embedding_layer = HiddenStateEmbedding(self.nmt_model)



        lstm_layers = []
        for i in range(7):
            lstm_layer = torch.nn.LSTM(512, embedding_size, batch_first=True, bidirectional=True).to("cuda")
            lstm_layers.append(lstm_layer)

        prob_entropy_lstm_layer = torch.nn.LSTM(2, 128, bidirectional=True)

        final_layers = get_feed_forward_layers(config["feed_forward_layers"]["dims"],
                                               config["feed_forward_layers"]["activation_function"],
                                               config["feed_forward_layers"]["activation_function_last_layer"],
                                               config["dropout"],
                                               )


        initialize_optimizer = get_optimizer_function(config)
        self.model = FullDecUtilityBaseModel(embedding_layer, lstm_layers, prob_entropy_lstm_layer, final_layers, utility,
                                             initialize_optimizer)
        return self.model
