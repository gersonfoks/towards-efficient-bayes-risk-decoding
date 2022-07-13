from utilities.preprocess.Preprocess import Preprocessor, HypToRefs, AddHypIds, Explode, GetProbEntropyLookupTable, \
    ResetIndex


class PreprocessFactory:

    def __init__(self, config, additional_arguments):
        self.config = config
        self.additional_arguments = additional_arguments

    def get_preprocessor(self):

        if self.config["name"] == "refs_with_prob_entropy":
            preprocessing_functions = [
                ResetIndex(),
                HypToRefs(),
                AddHypIds(),
                Explode(),

            ]

            table_functions = [
                GetProbEntropyLookupTable(
                    self.additional_arguments["wrapped_nmt_model"],
                    table_location=self.additional_arguments["table_location"]
                )

            ]
            preprocessor = Preprocessor(preprocessing_functions, table_functions)

            return preprocessor
        else:
            raise ValueError("preprocessor not found")
