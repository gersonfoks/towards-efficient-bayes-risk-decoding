from models.reference_models.FullDecCometModel.Collator import FullDecCometModelCollator
from models.reference_models.FullDecCometModel.Preprocess import FullDecCometModelPreprocess
from models.reference_models.FullDecCometModel.manager import FullDecCometBaseManager
from models.reference_models.FullDecCometModel.model import FullDecCometModel


class FullDecCometModelInfo:
    manager = FullDecCometBaseManager
    model = FullDecCometModel
    preprocess = FullDecCometModelPreprocess
    collator = FullDecCometModelCollator
