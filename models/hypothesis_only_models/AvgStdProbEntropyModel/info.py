from models.hypothesis_only_models.AvgStdProbEntropyModel.Collator import AvgStdProbEntropyModelCollator
from models.hypothesis_only_models.AvgStdProbEntropyModel.Manager import AvgStdProbEntropyModelManager
from models.hypothesis_only_models.AvgStdProbEntropyModel.Preprocess import AvgStdProbEntropyModelPreprocess
from models.hypothesis_only_models.AvgStdProbEntropyModel.model import AvgStdProbEntropyModel


class AvgStdProbEntropyModelInfo:

    manager = AvgStdProbEntropyModelManager
    model = AvgStdProbEntropyModel
    preprocess = AvgStdProbEntropyModelPreprocess
    collate = AvgStdProbEntropyModelCollator