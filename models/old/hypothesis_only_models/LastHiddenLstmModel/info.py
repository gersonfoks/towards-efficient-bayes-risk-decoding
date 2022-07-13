

# This class contains all the information for the training and hyperparam search
# It aids reusability.
from models.old.hypothesis_only_models import LastHiddenLstmModel
from models.old.hypothesis_only_models.LastHiddenLstmModel.Collator import LastHiddenLstmCollator
from models.old.hypothesis_only_models.LastHiddenLstmModel.Preprocess import LastHiddenLstmPreprocess
from models.old.hypothesis_only_models.LastHiddenLstmModel.manager import LastHiddenLstmManager


class LastHiddenStateLstmModelInfo:

    model_type = "hidden_state_model"
    manager = LastHiddenLstmManager
    model = LastHiddenLstmModel
    preprocess = LastHiddenLstmPreprocess
    collate = LastHiddenLstmCollator