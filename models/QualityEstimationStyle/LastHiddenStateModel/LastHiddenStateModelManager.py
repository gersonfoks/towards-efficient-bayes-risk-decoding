
from models.QualityEstimationStyle.LastHiddenStateModel.LastHiddenStateModel import LastHiddenStateModel
from models.common.layers.embedding import LastStateEmbedding
from models.common.layers.helpers import get_feed_forward_layers

from models.common.layers.pooling import LstmPoolingLayer, LearnedPoolingLayer
from models.common.optimization import get_optimizer_function
from models.base.BaseManager import BaseManager
from utilities.misc import load_nmt_model



class LastHiddenStateModelManager(BaseManager):

    def __init__(self, config, nmt_model=None, tokenizer=None):
        super().__init__(config)
        self.config = config

        self.nmt_model = nmt_model
        self.tokenizer = tokenizer

    def create_model(self):
        config = self.config
        self.nmt_model, self.tokenizer = load_nmt_model(config["nmt_model"], pretrained=True)

        # Create the embedding layer

        embedding_layer = LastStateEmbedding(self.nmt_model)


        pooling = None
        # Next we choose a way of pooling
        if config["pooling"]["name"] == "lstm":

            pooling = LstmPoolingLayer(config["pooling"]["embedding_size"], config["pooling"]["hidden_state_size"])
        elif config["pooling"]["name"] == "attention":
            pooling = LearnedPoolingLayer(config["pooling"]["embedding_size"],config["pooling"]["n_queries"],  config["pooling"]["n_heads"])
        else:
            raise ValueError('Unknown pooling: ', config["pooling"]["name"])


        final_layers = get_feed_forward_layers(config["feed_forward_layers"]["dims"],
                                               config["feed_forward_layers"]["activation_function"],
                                               config["feed_forward_layers"]["activation_function_last_layer"],
                                               config["dropout"],
                                               last_layer_scale=config["feed_forward_layers"]['last_layer_scale']
                                               )

        initialize_optimizer = get_optimizer_function(config)

        self.model = LastHiddenStateModel(embedding_layer, pooling, final_layers, initialize_optimizer)
        return self.model

