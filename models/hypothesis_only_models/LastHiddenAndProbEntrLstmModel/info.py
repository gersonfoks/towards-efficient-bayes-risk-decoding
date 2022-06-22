# This class contains all the information for the training and hyperparam search
# It aids reusability.
from models.hypothesis_only_models import LastHiddenLstmModel
from models.hypothesis_only_models.LastHiddenAndProbEntrLstmModel.Collator import LastHiddenProbEntrLstmCollator
from models.hypothesis_only_models.LastHiddenAndProbEntrLstmModel.manager import LastHiddenAndProbEntrLstmBaseManager
from models.hypothesis_only_models.LastHiddenAndProbEntrLstmModel.model import \
    LastHiddenAndProbEntrLstmModel
from models.hypothesis_only_models.LastHiddenAndProbEntrLstmModel.Preprocess import \
    LastHiddenAndProbEntrLstmModelPreprocess

class LastHiddenAndProbEntrLstmModelInfo:
    manager = LastHiddenAndProbEntrLstmBaseManager
    model = LastHiddenAndProbEntrLstmModel
    preprocess = LastHiddenAndProbEntrLstmModelPreprocess
    collate = LastHiddenProbEntrLstmCollator
