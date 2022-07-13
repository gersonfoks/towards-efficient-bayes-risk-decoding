import torch

from models.common.layers import get_feed_forward_layers
from models.common.optimization import get_optimizer_function

from models.Base.BaseManager import BaseManager
from models.old.hypothesis_only_models.TopProbModelV2.model import ProbSumModelV2

from utilities.misc import load_nmt_model


class TopProbModelV2Manager(BaseManager):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def create_model(self):
        config = self.config
        self.nmt_model, self.tokenizer = load_nmt_model(config["nmt_model"], pretrained=True)

        h_gru = torch.nn.GRU(1, 64, batch_first=True, bidirectional=True)
        top_k_gru = torch.nn.GRU(8, 256, batch_first=True, bidirectional=True)


        final_layers = get_feed_forward_layers(config["feed_forward_layers"]["dims"],
                                               config["feed_forward_layers"]["activation_function"],
                                               config["feed_forward_layers"]["activation_function_last_layer"],
                                               config["dropout"],
                                               )

        initialize_optimizer = get_optimizer_function(config)
        self.model = ProbSumModelV2( h_gru, top_k_gru, final_layers, initialize_optimizer)
        return self.model
