from collators.BasicCollator import BasicCollator
from collators.FullDecCollator import FullDecCollator
from collators.NMTCollator import NMTCollator
from collators.RefFullDecCollator import RefFullDecCollator
from collators.TokenStatisticsCollator import TokenStatisticsCollator


class CollatorFactory:

    def __init__(self, config, wrapped_nmt_model, tables=None):
        self.config = config
        self.wrapped_nmt_model = wrapped_nmt_model
        self.tables = tables

    def get_collators(self):

        if self.config["name"] == "basic_collator":

            return BasicCollator(self.wrapped_nmt_model.tokenizer), BasicCollator(self.wrapped_nmt_model.tokenizer)
        elif self.config["name"] == "nmt_collator":
            return NMTCollator(self.wrapped_nmt_model.nmt_model, self.wrapped_nmt_model.tokenizer), NMTCollator(
                self.wrapped_nmt_model.nmt_model, self.wrapped_nmt_model.tokenizer)
        elif self.config["name"] == "token_statics_collator":
            return TokenStatisticsCollator(self.tables[0]), TokenStatisticsCollator(self.tables[1])
        elif self.config["name"] == "full_dec_collator":
            return FullDecCollator(self.wrapped_nmt_model.nmt_model, self.wrapped_nmt_model.tokenizer, self.tables[0]), \
                   FullDecCollator(self.wrapped_nmt_model.nmt_model, self.wrapped_nmt_model.tokenizer, self.tables[1])
        elif self.config["name"] == "ref_full_dec_collator":
            return RefFullDecCollator(self.wrapped_nmt_model.nmt_model, self.wrapped_nmt_model.tokenizer, self.tables[0]), \
                   RefFullDecCollator(self.wrapped_nmt_model.nmt_model, self.wrapped_nmt_model.tokenizer, self.tables[1])
        else:
            raise ValueError("collator not known: ", self.config["name"])
