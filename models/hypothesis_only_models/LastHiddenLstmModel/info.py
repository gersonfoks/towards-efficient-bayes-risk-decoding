

# This class contains all the information for the training and hyperparam search
# It aids reusability.
from models.hypothesis_only_models import LastHiddenLstmModel
from models.hypothesis_only_models.LastHiddenLstmModel.Collator import LastHiddenLstmCollator
from models.hypothesis_only_models.LastHiddenLstmModel.Preprocess import LastHiddenLstmPreprocess
from models.hypothesis_only_models.LastHiddenLstmModel.manager import LastHiddenLstmManager


class LastHiddenStateLstmModelInfo:

    manager = LastHiddenLstmManager
    model = LastHiddenLstmModel
    preprocess = LastHiddenLstmPreprocess
    collate = LastHiddenLstmCollator