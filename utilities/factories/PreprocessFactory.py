from utilities.preprocess.Preprocess import Preprocessor, HypToRefs, AddHypIds, Explode, ResetIndex, UtilitiesToAverage, \
    AddRefUtilities


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
            preprocessor = Preprocessor(preprocessing_functions)

            return preprocessor
        elif self.config["name"] == "refs_full_dec":
            preprocessing_functions = [
                ResetIndex(),
                AddRefUtilities(),
                UtilitiesToAverage(),
                HypToRefs(),
                AddHypIds(),
                Explode(cols=["hypotheses", "utilities", "count", "hypotheses_ids", "ref_utilities"]),
                ResetIndex(),

            ]
            preprocessor = Preprocessor(preprocessing_functions)

            return preprocessor
        elif self.config["name"] == "basic":
            preprocessing_functions = [
                ResetIndex(),
                UtilitiesToAverage(),
                Explode(cols=["hypotheses", "utilities", "count"]),
                ResetIndex(),
            ]


            preprocessor = Preprocessor(preprocessing_functions)

            return preprocessor

        else:
            raise ValueError("preprocessor not found")
