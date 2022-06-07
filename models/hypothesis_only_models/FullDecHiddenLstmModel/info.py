

# This class contains all the information for the training and hyperparam search
# It aids reusability.
from models.hypothesis_only_models import LastHiddenLstmModel
from models.hypothesis_only_models.FullDecHiddenLstmModel.Collator import FullDecHiddenLstmModelCollator
from models.hypothesis_only_models.FullDecHiddenLstmModel.Preprocess import FullDecHiddenLstmModelPreprocess
from models.hypothesis_only_models.FullDecHiddenLstmModel.manager import FullDecHiddenLstmModelManager
from models.hypothesis_only_models.FullDecHiddenLstmModel.model import FullDecHiddenLstmModel
from models.hypothesis_only_models.LastHiddenLstmModel.Collator import LastHiddenLstmCollator
from models.hypothesis_only_models.LastHiddenLstmModel.Preprocess import LastHiddenLstmPreprocess
from models.hypothesis_only_models.LastHiddenLstmModel.manager import LastHiddenLstmManager


class FullDecHiddenLstmInfo:

    manager = FullDecHiddenLstmModelManager
    model = FullDecHiddenLstmModel
    preprocess = FullDecHiddenLstmModelPreprocess
    collate = FullDecHiddenLstmModelCollator