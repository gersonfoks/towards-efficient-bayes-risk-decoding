from models.old.reference_models.CometEncoddingModel.Collator import CometEncodingModelCollator
from models.old.reference_models.CometEncoddingModel.Preprocess import CometEncodingModelPreprocess
from models.old.reference_models.CometEncoddingModel.manager import CometEncodingBaseManager
from models.old.reference_models.CometEncoddingModel.model import CometEncodingModel


class CometEncodingModelInfo:


    manager = CometEncodingBaseManager
    model = CometEncodingModel
    preprocess = CometEncodingModelPreprocess
    collator = CometEncodingModelCollator