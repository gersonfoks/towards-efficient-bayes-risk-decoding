import torch
from torch import nn

from models.common.layers import get_feed_forward_layers, GlobalMeanPooling, GlobalMaxPooling, LearnedPoolingLayer, \
    AttentionWithLearnableEmbedding
from models.common.optimization import get_optimizer_function

from models.Base.BaseManager import BaseManager


from models.reference_models.CrossAttentionModelV3.model import CrossAttentionModelV3
from models.reference_models.CrossAttentionModelV4.model import CrossAttentionModelV4
from utilities.misc import load_nmt_model


class CrossAttentionModelV4Manager(BaseManager):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def create_model(self):
        config = self.config
        self.nmt_model, self.tokenizer = load_nmt_model(config["nmt_model"], pretrained=True)

        # Create the embedding layer

        prob_entrop_embed_size = config["prob_entrop_embed_size"]

        embedding_layer = self.nmt_model.get_decoder().get_input_embeddings()
        prob_entropy_embedding_layer = nn.Sequential(nn.Linear(2, prob_entrop_embed_size))

        n_heads = self.config["n_heads"]
        n_learnable_embeddings = self.config["n_learnable_embeddings"]

        cross_attention = AttentionWithLearnableEmbedding(512, 512, prob_entrop_embed_size, n_learnable_embeddings, n_heads)
        self_attention = AttentionWithLearnableEmbedding(512, 512, prob_entrop_embed_size, n_learnable_embeddings, n_heads)


        weights_layer = get_feed_forward_layers(config["weights_layers"]["dims"],
                                               config["weights_layers"]["activation_function"],
                                               config["weights_layers"]["activation_function_last_layer"],
                                               config["dropout"], batch_norm=self.config["batch_norm"],
                                               )


        h_pred_layer = get_feed_forward_layers(config["pred_layers"]["dims"],
                                               config["pred_layers"]["activation_function"],
                                               config["pred_layers"]["activation_function_last_layer"],
                                               config["dropout"], batch_norm=self.config["batch_norm"],
                                               )

        ref_pred_layer = get_feed_forward_layers(config["pred_layers"]["dims"],
                                               config["pred_layers"]["activation_function"],
                                               config["pred_layers"]["activation_function_last_layer"],
                                               config["dropout"], batch_norm=self.config["batch_norm"],
                                               )

        initialize_optimizer = get_optimizer_function(config)
        self.model = CrossAttentionModelV4(embedding_layer, prob_entropy_embedding_layer, self_attention,
                                           cross_attention,
                                           weights_layer, h_pred_layer, ref_pred_layer, initialize_optimizer)
        return self.model
