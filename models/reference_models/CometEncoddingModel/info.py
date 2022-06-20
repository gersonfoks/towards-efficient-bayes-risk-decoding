from models.reference_models.CometEncoddingModel.Collator import CometEncodingModelCollator
from models.reference_models.CometEncoddingModel.Preprocess import CometEncodingModelPreprocess
from models.reference_models.CometEncoddingModel.manager import CometEncodingModelManager
from models.reference_models.CometEncoddingModel.model import CometEncodingModel


class CometEncodingModelInfo:


    manager = CometEncodingModelManager
    model = CometEncodingModel
    preprocess = CometEncodingModelPreprocess
    collator = CometEncodingModelCollator