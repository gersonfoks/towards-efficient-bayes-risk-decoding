from models.old.hypothesis_only_models.AvgStdProbEntropyModel.Collator import AvgStdProbEntropyModelCollator
from models.old.hypothesis_only_models.AvgStdProbEntropyModel.Manager import AvgStdProbEntropyBaseManager
from models.old.hypothesis_only_models.AvgStdProbEntropyModel.Preprocess import AvgStdProbEntropyModelPreprocess
from models.old.hypothesis_only_models.AvgStdProbEntropyModel.model import AvgStdProbEntropyModel


class AvgStdProbEntropyModelInfo:

    manager = AvgStdProbEntropyBaseManager
    model = AvgStdProbEntropyModel
    preprocess = AvgStdProbEntropyModelPreprocess
    collate = AvgStdProbEntropyModelCollator