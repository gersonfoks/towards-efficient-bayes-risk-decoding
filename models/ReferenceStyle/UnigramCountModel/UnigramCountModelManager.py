import torch.nn

from models.ReferenceStyle.UnigramCountModel.UnigramCountModel import UnigramCountModel
from models.common.layers.embedding import UnigramCountEmbedding

from models.common.layers.helpers import get_feed_forward_layers

from models.common.optimization import get_optimizer_function
from models.base.BaseManager import BaseManager
from utilities.misc import load_nmt_model


class UnigramCountModelManager(BaseManager):

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

        # We use the pretrained embeddings.
        embedding = self.nmt_model.get_decoder().get_input_embeddings()
        unigram_count_embedding = UnigramCountEmbedding(embedding, padding_id=58100)
        attention_layer = torch.nn.MultiheadAttention(512, config["n_heads"], batch_first=True)

        final_layers = get_feed_forward_layers(config["feed_forward_layers"]["dims"],
                                               config["feed_forward_layers"]["activation_function"],
                                               config["feed_forward_layers"]["activation_function_last_layer"],
                                               config["dropout"],
                                               last_layer_scale=config["feed_forward_layers"]['last_layer_scale']
                                               )

        initialize_optimizer = get_optimizer_function(config)

        self.model = UnigramCountModel(unigram_count_embedding, attention_layer, final_layers, initialize_optimizer,
                                       config["min_value"],
                                       config["max_value"],
                                       )
        return self.model
