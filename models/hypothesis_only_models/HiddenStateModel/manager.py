import torch

from models.common.layers import EmbbedingForPackedSequenceLayer, get_feed_forward_layers, LastStateEmbedding, \
    HiddenStateEmbedding
from models.common.optimization import get_optimizer_function
from models.hypothesis_only_models.HiddenStateModel.HiddenStateModel import HiddenStateModel
from models.hypothesis_only_models.HypothesisLstmModel.model import HypothesisLstmModel
from models.hypothesis_only_models.LastHiddenLstmModel.LastHiddenLstmModel import LastHiddenLstmModel
from models.manager import ModelManager
from utilities.misc import load_nmt_model
from pathlib import Path

class HiddenStateModelManager(ModelManager):



    def create_model(self):
        config = self.config
        self.nmt_model, self.tokenizer = load_nmt_model(config["nmt_model"], pretrained=True)

        # Create the embedding layer


        embedding_layer = HiddenStateEmbedding(self.nmt_model)


        final_layers = get_feed_forward_layers(config["feed_forward_layers"]["dims"],
                                               config["feed_forward_layers"]["activation_function"],
                                               config["feed_forward_layers"]["activation_function_last_layer"],
                                               config["dropout"],
                                               )

        initialize_optimizer = get_optimizer_function(config)
        self.model = HiddenStateModel(embedding_layer, final_layers, initialize_optimizer)
        return self.model


