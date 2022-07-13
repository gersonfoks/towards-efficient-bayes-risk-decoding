from models.old.reference_models.BasicReferenceLstmModelV2.Collator import BasicReferenceLstmModelV2Collator
from models.old.reference_models.BasicReferenceLstmModelV2.Preprocess import BasicReferenceLstmModelV2Preprocess
from models.old.reference_models.BasicReferenceLstmModelV2.manager import BasicReferenceLstmBaseV2Manager
from models.old.reference_models.BasicReferenceLstmModelV2.model import BasicReferenceLstmModelV2


class BasicReferenceLstmModelV2Info:


    manager = BasicReferenceLstmBaseV2Manager
    model = BasicReferenceLstmModelV2
    preprocess = BasicReferenceLstmModelV2Preprocess
    collator = BasicReferenceLstmModelV2Collator