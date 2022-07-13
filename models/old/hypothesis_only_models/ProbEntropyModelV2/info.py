

# This class contains all the information for the training and hyperparam search
# It aids reusability.
from models.old.hypothesis_only_models.ProbEntropyModelV2.collator import ProbEntropyModelCollatorV2
from models.old.hypothesis_only_models.ProbEntropyModelV2.manager import ProbEntropyBaseManagerV2
from models.old.hypothesis_only_models.ProbEntropyModelV2.model import ProbEntropyModelV2
from models.old.hypothesis_only_models.ProbEntropyModelV2.preprocess import ProbEntropyModelPreprocessV2


class ProbEntropyModelInfoV2:

    manager = ProbEntropyBaseManagerV2
    model = ProbEntropyModelV2
    preprocess = ProbEntropyModelPreprocessV2
    collate = ProbEntropyModelCollatorV2