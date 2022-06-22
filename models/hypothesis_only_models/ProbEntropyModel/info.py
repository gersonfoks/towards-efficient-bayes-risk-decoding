

# This class contains all the information for the training and hyperparam search
# It aids reusability.
from models.hypothesis_only_models.ProbEntropyModel.collator import ProbEntropyModelCollator
from models.hypothesis_only_models.ProbEntropyModel.manager import ProbEntropyBaseManager
from models.hypothesis_only_models.ProbEntropyModel.model import ProbEntropyModel
from models.hypothesis_only_models.ProbEntropyModel.preprocess import ProbEntropyModelPreprocess


class ProbEntropyModelInfo:

    manager = ProbEntropyBaseManager
    model = ProbEntropyModel
    preprocess = ProbEntropyModelPreprocess
    collate = ProbEntropyModelCollator