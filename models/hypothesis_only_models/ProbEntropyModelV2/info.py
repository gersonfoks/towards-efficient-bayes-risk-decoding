

# This class contains all the information for the training and hyperparam search
# It aids reusability.
from models.ProbEntropyModelV2.collator import ProbEntropyModelCollatorV2
from models.ProbEntropyModelV2.manager import ProbEntropyModelManagerV2
from models.ProbEntropyModelV2.model import ProbEntropyModelV2
from models.ProbEntropyModelV2.preprocess import ProbEntropyModelPreprocessV2
from models.hypothesis_only_models.ProbEntropyModel.collator import ProbEntropyModelCollator
from models.hypothesis_only_models.ProbEntropyModel.manager import ProbEntropyModelManager
from models.hypothesis_only_models.ProbEntropyModel.model import ProbEntropyModel
from models.hypothesis_only_models.ProbEntropyModel.preprocess import ProbEntropyModelPreprocess


class ProbEntropyModelInfoV2:

    manager = ProbEntropyModelManagerV2
    model = ProbEntropyModelV2
    preprocess = ProbEntropyModelPreprocessV2
    collate = ProbEntropyModelCollatorV2