'''
This file contains an abstract class for the model managers
'''
from pathlib import Path

import torch


class BaseManager:


    def __init__(self, config):
        self.model = None
        self.config = config


    def create_model(self):
        raise NotImplementedError()

    def save_model(self, save_model_path):
        Path(save_model_path).mkdir(parents=True, exist_ok=True)
        pl_path = save_model_path + 'pl_model.pt'

        state = {
            "config": self.config,
            "state_dict": self.model.state_dict()
        }

        torch.save(state, pl_path)

    @classmethod
    def load_model(cls, model_path):
        '''
        Returns a manager and a loaded model specified by the file pointed by the model_path
        :param model_path:
        :return:
        '''

        pl_path = model_path + 'pl_model.pt'
        checkpoint = torch.load(pl_path)
        manager = cls(checkpoint["config"])
        model = manager.create_model()

        model.load_state_dict(checkpoint["state_dict"])
        return model, manager