from models.old.reference_models.BasicReferenceLstmModel.Collator import BasicReferenceLstmModelCollator
from models.old.reference_models.BasicReferenceLstmModel.Preprocess import BasicReferenceLstmModelPreprocess
from models.old.reference_models.BasicReferenceLstmModel.manager import BasicReferenceLstmBaseManager
from models.old.reference_models.BasicReferenceLstmModel.model import BasicReferenceLstmModel


class BasicReferenceLstmModelInfo:


    manager = BasicReferenceLstmBaseManager
    model = BasicReferenceLstmModel
    preprocess = BasicReferenceLstmModelPreprocess
    collator = BasicReferenceLstmModelCollator