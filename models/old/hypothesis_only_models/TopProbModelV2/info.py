from models.old.hypothesis_only_models import TopProbModelV2
from models.old.hypothesis_only_models.TopProbModel.Preprocess import TopProbModelPreprocess
from models.old.hypothesis_only_models.TopProbModelV2.Collator import TopProbModelV2Collator
from models.old.hypothesis_only_models.TopProbModelV2.manager import TopProbModelV2Manager


class TopProbModelV2Info:


    manager = TopProbModelV2Manager
    model = TopProbModelV2
    preprocess = TopProbModelPreprocess
    collator = TopProbModelV2Collator