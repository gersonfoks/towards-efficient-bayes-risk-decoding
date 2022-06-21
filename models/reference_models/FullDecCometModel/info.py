from models.reference_models.FullDecCometModel.Collator import FullDecCometModelCollator
from models.reference_models.FullDecCometModel.Preprocess import FullDecCometModelPreprocess
from models.reference_models.FullDecCometModel.manager import FullDecCometModelManager
from models.reference_models.FullDecCometModel.model import FullDecCometModel


class FullDecCometModelInfo:
    manager = FullDecCometModelManager
    model = FullDecCometModel
    preprocess = FullDecCometModelPreprocess
    collator = FullDecCometModelCollator
