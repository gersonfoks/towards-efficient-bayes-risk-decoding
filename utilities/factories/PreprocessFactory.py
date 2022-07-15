from utilities.preprocess.Preprocess import Preprocessor, HypToRefs, AddHypIds, Explode, ResetIndex, UtilitiesToAverage


class PreprocessFactory:

    def __init__(self, config, additional_arguments=None):
        self.config = config
        self.additional_arguments = additional_arguments

    def get_preprocessor(self):

        if self.config["name"] == "refs_with_prob_entropy":
            preprocessing_functions = [
                ResetIndex(),
                UtilitiesToAverage(),
                HypToRefs(),
                AddHypIds(),
                Explode(),
                ResetIndex(),

            ]

            # table_functions = [
            #     GetProbEntropyLookupTable(
            #         self.additional_arguments["wrapped_nmt_model"],
            #         table_location=self.additional_arguments["table_location"]
            #     )

            preprocessor = Preprocessor(preprocessing_functions)

            return preprocessor
        elif self.config["name"] == "basic":
            preprocessing_functions = [
                ResetIndex(),
                UtilitiesToAverage(),
                Explode(cols=["hypotheses", "utilities", "count"]),
                ResetIndex(),
            ]

            table_functions = []
            preprocessor = Preprocessor(preprocessing_functions)

            return preprocessor

        else:
            raise ValueError("preprocessor not found")
