import torch.nn
from torch import nn

from models.QualityEstimationStyle.FullDecModel.FullDecModel import FullDecModel
from models.ReferenceStyle.CometAttentionModel.CometAttentionModel import CometAttentionModel
from models.ReferenceStyle.ReferenceFullDecModel.ReferenceFullDecModel import ReferenceFullDecModel

from models.common.layers.embedding import LastStateEmbedding, HiddenStateEmbedding, CometEmbedding
from models.common.layers.helpers import get_feed_forward_layers

from models.common.layers.pooling import LstmPoolingLayer, LearnedPoolingLayer, FullDecPooling
from models.common.optimization import get_optimizer_function
from models.Base.BaseManager import BaseManager
from utilities.misc import load_nmt_model, load_comet_model


class CometAttentionModelManager(BaseManager):

    def __init__(self, config, nmt_model=None, tokenizer=None, comet_model=None):
        super().__init__(config)
        self.config = config

        self.nmt_model = nmt_model
        self.tokenizer = tokenizer
        self.comet_model = comet_model

    def create_model(self):
        config = self.config

        if self.nmt_model == None:
            self.nmt_model, self.tokenizer = load_nmt_model(config["nmt_model"], pretrained=True)

        if self.comet_model == None:
            comet_model = load_comet_model()

        # Create the embedding layer

        embedding_layer = HiddenStateEmbedding(self.nmt_model)

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

        token_embedding_layer = nn.Linear(config["n_statistics"], config["embedding_size"])

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

        full_dec_pooling = FullDecPooling(embedding_layer, token_embedding_layer, full_dec_pooling_layers,
                                          token_statistics_pooling)

        nmt_down = nn.Linear(4096, 512)

        comet_embedding = CometEmbedding(
            self.comet_model, nn.Linear(6 * 512, 512),
        )

        nmt_comet_attention = nn.MultiheadAttention(512, 4)

        final_layers = get_feed_forward_layers(config["feed_forward_layers"]["dims"],
                                               config["feed_forward_layers"]["activation_function"],
                                               config["feed_forward_layers"]["activation_function_last_layer"],
                                               config["dropout"],
                                               last_layer_scale=config["feed_forward_layers"]['last_layer_scale']
                                               )

        # Initialize the gate to 0.5

        initialize_optimizer = get_optimizer_function(config)

        self.model = CometAttentionModel(full_dec_pooling, nmt_down, comet_embedding, nmt_comet_attention, final_layers,
                                         initialize_optimizer)
        return self.model
