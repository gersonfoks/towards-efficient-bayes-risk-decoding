'''
This file contains a abstract class for the model managers
'''


class ModelManager:

    def create_model(self):
        raise NotImplementedError()


    def save_model(self, save_model_path):
        raise NotImplementedError()


    def load_model(self, model_path):
        raise NotImplementedError()