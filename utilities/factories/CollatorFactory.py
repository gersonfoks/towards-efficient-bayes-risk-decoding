from collators.BasicCollator import BasicCollator


class CollatorFactory:

    def __init__(self, config, wrapped_nmt_model):
        self.config = config
        self.wrapped_nmt_model = wrapped_nmt_model


    def get_collators(self):

        if self.config["name"] == "basic_collator":

            return BasicCollator(self.wrapped_nmt_model.tokenizer), BasicCollator(self.wrapped_nmt_model.tokenizer)
        else:
            raise ValueError("collator not known: ", self.config["name"])
