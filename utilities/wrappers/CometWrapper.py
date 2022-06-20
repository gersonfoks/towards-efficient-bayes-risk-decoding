import torch



class CometWrapper:

    def __init__(self, cometModel, device='cuda'):
        super().__init__()

        self.model = cometModel

        self.device = device


    def to_embedding(self, strings):
        inputs = self.model.encoder.prepare_sample(strings).to(self.device)

        return self.model.get_sentence_embedding(**inputs)
