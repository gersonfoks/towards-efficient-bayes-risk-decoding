from comet import download_model, load_from_checkpoint
from torch import nn

from models.QualityEstimationStyle.CometModel.CometModel import CometModel
from models.QualityEstimationStyle.FullDecModel.FullDecModel import FullDecModel

from models.common.layers.embedding import  FullDecEmbedding
from models.common.layers.helpers import get_feed_forward_layers

from models.common.layers.pooling import LstmPoolingLayer, LearnedPoolingLayer
from models.common.optimization import get_optimizer_function
from models.base.BaseManager import BaseManager
from utilities.misc import load_nmt_model
from utilities.wrappers.CometWrapper import CometWrapper


class CometModelManager(BaseManager):

    def __init__(self, config, nmt_model=None, tokenizer=None):
        super().__init__(config)
        self.config = config

        self.nmt_model = nmt_model
        self.tokenizer = tokenizer

    def create_model(self):
        config = self.config


        # Load the comet model and put it inside a wrapper
        model_path = download_model("wmt20-comet-da")
        model = load_from_checkpoint(model_path)
        model.to("cuda")
        model.eval()
        wrapped_model = CometWrapper(model)


        final_layers = get_feed_forward_layers(config["feed_forward_layers"]["dims"],
                                               config["feed_forward_layers"]["activation_function"],
                                               config["feed_forward_layers"]["activation_function_last_layer"],
                                               config["dropout"],
                                               last_layer_scale=config["feed_forward_layers"]['last_layer_scale']
                                               )

        initialize_optimizer = get_optimizer_function(config)

        self.model = CometModel(wrapped_model, final_layers, initialize_optimizer)
        return self.model
