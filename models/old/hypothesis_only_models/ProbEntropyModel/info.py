

# This class contains all the information for the training and hyperparam search
# It aids reusability.
from models.old.hypothesis_only_models.ProbEntropyModel.collator import ProbEntropyModelCollator
from models.old.hypothesis_only_models.ProbEntropyModel.manager import ProbEntropyBaseManager
from models.old.hypothesis_only_models.ProbEntropyModel.model import ProbEntropyModel
from models.old.hypothesis_only_models.ProbEntropyModel.preprocess import ProbEntropyModelPreprocess


class ProbEntropyModelInfo:

    manager = ProbEntropyBaseManager
    model = ProbEntropyModel
    preprocess = ProbEntropyModelPreprocess
    collate = ProbEntropyModelCollator