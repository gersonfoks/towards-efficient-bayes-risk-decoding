from torch import nn

from models.MixtureModels.FullDecMixtureModel import FullDecMixtureModel

from models.common.layers.embedding import FullDecEmbedding
from models.common.layers.helpers import get_feed_forward_layers, ScaledActivationFunction

from models.common.layers.pooling import LstmPoolingLayer, LearnedPoolingLayer
from models.common.optimization import get_optimizer_function
from models.base.BaseManager import BaseManager
from utilities.misc import load_nmt_model


class FullDecMixtureModelManager(BaseManager):

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

        embedding_layer = FullDecEmbedding(self.nmt_model, config["token_statistics_embedding_size"])

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

        token_statistics_pooling = None
        # Next we choose a way of pooling
        if config["token_pooling"]["name"] == "lstm":

            token_statistics_pooling = LstmPoolingLayer(config["token_pooling"]["embedding_size"],
                                                        config["token_pooling"]["hidden_state_size"])
        elif config["token_pooling"]["name"] == "attention":
            token_statistics_pooling = LearnedPoolingLayer(config["token_pooling"]["embedding_size"],
                                                           config["token_pooling"]["n_heads"])
        else:
            raise ValueError('Unknown pooling: ', config["token_pooling"]["name"])

        final_layers = get_feed_forward_layers(config["feed_forward_layers"]["dims"],
                                               config["feed_forward_layers"]["activation_function"],
                                               None,
                                               config["dropout"],

                                               )

        initialize_optimizer = get_optimizer_function(config)

        locs_activation_function = ScaledActivationFunction(nn.Tanh(), 2.5)

        self.model = FullDecMixtureModel(embedding_layer, full_dec_pooling_layers,
                                         token_statistics_pooling, final_layers, initialize_optimizer,
                                         locs_activation_function, n_components=config["n_components"])
        return self.model
