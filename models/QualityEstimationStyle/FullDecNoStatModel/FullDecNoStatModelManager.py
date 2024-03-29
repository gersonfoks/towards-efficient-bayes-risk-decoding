from torch import nn

from models.QualityEstimationStyle.FullDecNoStatModel.FullDecNoStatModel import FullDecNoStatModel

from models.common.layers.embedding import FullDecEmbedding
from models.common.layers.helpers import get_feed_forward_layers

from models.common.layers.pooling import LstmPoolingLayer, LearnedPoolingLayer
from models.common.optimization import get_optimizer_function
from models.base.BaseManager import BaseManager
from utilities.misc import load_nmt_model


class FullDecNoStatModelManager(BaseManager):

    def __init__(self, config, nmt_model=None, tokenizer=None):
        super().__init__(config)
        self.config = config

        self.nmt_model = nmt_model
        self.tokenizer = tokenizer

    def create_model(self):
        config = self.config

        if self.nmt_model == None:
            self.nmt_model, self.tokenizer = load_nmt_model(config["nmt_model"], pretrained=True)

        # Create the embedding layer

        embedding_layer = FullDecEmbedding(self.nmt_model,None)

        full_dec_pooling_layers = []

        for i in range(7):
            pooling = None
            if config["dec_pooling"]["name"] == "lstm":

                pooling = LstmPoolingLayer(config["dec_pooling"]["embedding_size"],
                                           config["dec_pooling"]["hidden_state_size"])
            elif config["dec_pooling"]["name"] == "attention":
                pooling = LearnedPoolingLayer(config["dec_pooling"]["embedding_size"], config["dec_pooling"]["n_heads"])
            full_dec_pooling_layers.append(pooling)

        full_dec_pooling_layers = nn.ModuleList(full_dec_pooling_layers)

        final_layers = get_feed_forward_layers(config["feed_forward_layers"]["dims"],
                                               config["feed_forward_layers"]["activation_function"],
                                               config["feed_forward_layers"]["activation_function_last_layer"],
                                               config["dropout"],
                                               last_layer_scale=config["feed_forward_layers"]['last_layer_scale']
                                               )

        initialize_optimizer = get_optimizer_function(config)

        self.model = FullDecNoStatModel(embedding_layer, full_dec_pooling_layers,
                                        final_layers, initialize_optimizer)
        return self.model
