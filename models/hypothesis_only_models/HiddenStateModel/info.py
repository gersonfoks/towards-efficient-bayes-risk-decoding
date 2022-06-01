from models.hypothesis_only_models.HiddenStateModel.Collator import HiddenStateModelCollator
from models.hypothesis_only_models.HiddenStateModel.HiddenStateModel import HiddenStateModel
from models.hypothesis_only_models.HiddenStateModel.Preprocess import HiddenStateModelPreprocess
from models.hypothesis_only_models.HiddenStateModel.manager import HiddenStateModelManager


# This class contains all the information for the training and hyperparam search
# It aids reusability.
class HiddenStateModelInfo:

    manager = HiddenStateModelManager
    model = HiddenStateModel
    preprocess = HiddenStateModelPreprocess
    collate = HiddenStateModelCollator