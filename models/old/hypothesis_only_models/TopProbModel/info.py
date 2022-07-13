from models.old.hypothesis_only_models import TopProbModel
from models.old.hypothesis_only_models.TopProbModel.Collator import TopProbModelCollator
from models.old.hypothesis_only_models.TopProbModel.Preprocess import TopProbModelPreprocess
from models.old.hypothesis_only_models.TopProbModel.manager import TopProbModelManager


class TopProbModelInfo:


    manager = TopProbModelManager
    model = TopProbModel
    preprocess = TopProbModelPreprocess
    collator = TopProbModelCollator