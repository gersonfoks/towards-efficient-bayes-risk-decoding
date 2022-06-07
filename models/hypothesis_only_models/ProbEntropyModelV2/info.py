

# This class contains all the information for the training and hyperparam search
# It aids reusability.
from models.hypothesis_only_models.ProbEntropyModelV2.collator import ProbEntropyModelCollatorV2
from models.hypothesis_only_models.ProbEntropyModelV2.manager import ProbEntropyModelManagerV2
from models.hypothesis_only_models.ProbEntropyModelV2.model import ProbEntropyModelV2
from models.hypothesis_only_models.ProbEntropyModelV2.preprocess import ProbEntropyModelPreprocessV2


class ProbEntropyModelInfoV2:

    manager = ProbEntropyModelManagerV2
    model = ProbEntropyModelV2
    preprocess = ProbEntropyModelPreprocessV2
    collate = ProbEntropyModelCollatorV2