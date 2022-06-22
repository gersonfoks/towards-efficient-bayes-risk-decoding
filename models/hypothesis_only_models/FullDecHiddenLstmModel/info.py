

from models.hypothesis_only_models.FullDecHiddenLstmModel.Collator import FullDecHiddenLstmModelCollator
from models.hypothesis_only_models.FullDecHiddenLstmModel.Preprocess import FullDecHiddenLstmModelPreprocess
from models.hypothesis_only_models.FullDecHiddenLstmModel.manager import FullDecHiddenLstmBaseManager
from models.hypothesis_only_models.FullDecHiddenLstmModel.model import FullDecHiddenLstmModel


class FullDecHiddenLstmInfo:

    manager = FullDecHiddenLstmBaseManager
    model = FullDecHiddenLstmModel
    preprocess = FullDecHiddenLstmModelPreprocess
    collate = FullDecHiddenLstmModelCollator