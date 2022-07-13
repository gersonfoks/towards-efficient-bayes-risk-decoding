import torch
from torch import nn


from models.common.layers import get_feed_forward_layers, LearnedPoolingLayer
from models.common.optimization import get_optimizer_function

from models.Base.BaseManager import BaseManager

from models.old.reference_models.CrossAttentionModelV2.model import CrossAttentionModelV2
from utilities.misc import load_nmt_model


class CrossAttentionModelV2Manager(BaseManager):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def create_model(self):
        config = self.config
        self.nmt_model, self.tokenizer = load_nmt_model(config["nmt_model"], pretrained=True)

        # Create the embedding layer

        embedding_size = config["embedding"]["size"]

        embedding_layer = self.nmt_model.get_decoder().get_input_embeddings()
        prob_entropy_embedding_layer = nn.Sequential(nn.Linear(2, 64))

        cross_attention = nn.MultiheadAttention(embedding_size, 4, batch_first=True, vdim=64)

        emb = torch.zeros(1, 2, 512) # Use two embeddings instead of one
        nn.init.xavier_normal(emb)
        learnable_embedding = torch.nn.Parameter(emb)
        emb = torch.zeros(1, 2, 64) # Use two embeddings instead of one
        nn.init.xavier_normal(emb)
        learnable_embedding_prob_entropy = torch.nn.Parameter(emb)



        pooling_layer = LearnedPoolingLayer(512,)

        final_layers = get_feed_forward_layers(config["feed_forward_layers"]["dims"],
                                               config["feed_forward_layers"]["activation_function"],
                                               config["feed_forward_layers"]["activation_function_last_layer"],
                                               config["dropout"],
                                               )

        initialize_optimizer = get_optimizer_function(config)
        self.model = CrossAttentionModelV2(embedding_layer, prob_entropy_embedding_layer, learnable_embedding, learnable_embedding_prob_entropy,  cross_attention, pooling_layer, final_layers, initialize_optimizer)
        return self.model
