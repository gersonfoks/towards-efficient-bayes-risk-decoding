from models.common.layers import get_feed_forward_layers
from models.common.optimization import get_optimizer_function

from models.Base.BaseManager import BaseManager

from models.reference_models.CometEncoddingModel.model import CometEncodingModel
from utilities.misc import load_nmt_model


class CometEncodingBaseManager(BaseManager):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def create_model(self):
        config = self.config
        self.nmt_model, self.tokenizer = load_nmt_model(config["nmt_model"], pretrained=True)

        # Create the embedding layer

        final_layers = get_feed_forward_layers(config["feed_forward_layers"]["dims"],
                                               config["feed_forward_layers"]["activation_function"],
                                               config["feed_forward_layers"]["activation_function_last_layer"],
                                               config["dropout"],
                                               )

        initialize_optimizer = get_optimizer_function(config)
        self.model = CometEncodingModel(final_layers, initialize_optimizer)
        return self.model
