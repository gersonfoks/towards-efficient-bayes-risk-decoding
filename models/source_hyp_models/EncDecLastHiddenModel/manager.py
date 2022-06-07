import torch

from models.common.layers import EmbbedingForPackedSequenceLayer, get_feed_forward_layers, LastStateEmbedding, \
    EncDecLastStateEmbedding
from models.common.optimization import get_optimizer_function

from models.manager import ModelManager
from models.source_hyp_models.EncDecLastHiddenModel.model import EncDecLastHiddenModel
from utilities.misc import load_nmt_model
from pathlib import Path


class EncDecLastHiddenModelManager(ModelManager):

    def create_model(self):
        config = self.config
        self.nmt_model, self.tokenizer = load_nmt_model(config["nmt_model"], pretrained=True)

        # Create the embedding layer

        embedding_size = 512

        embedding_layer = EncDecLastStateEmbedding(self.nmt_model)

        enc_lstm_layer = torch.nn.LSTM(embedding_size, embedding_size, batch_first=True, bidirectional=True)
        dec_lstm_layer = torch.nn.LSTM(embedding_size, embedding_size, batch_first=True, bidirectional=True)

        final_layers = get_feed_forward_layers(config["feed_forward_layers"]["dims"],
                                               config["feed_forward_layers"]["activation_function"],
                                               config["feed_forward_layers"]["activation_function_last_layer"],
                                               config["dropout"],
                                               )

        initialize_optimizer = get_optimizer_function(config)
        self.model = EncDecLastHiddenModel(embedding_layer, enc_lstm_layer, dec_lstm_layer, final_layers, initialize_optimizer)
        return self.model

    def save_model(self, save_model_path):
        Path(save_model_path).mkdir(parents=True, exist_ok=True)
        pl_path = save_model_path + 'pl_model.pt'

        state = {
            "config": self.config,
            "state_dict": self.model.state_dict()
        }

        torch.save(state, pl_path)
