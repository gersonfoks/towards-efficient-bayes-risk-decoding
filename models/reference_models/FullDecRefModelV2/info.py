from models.reference_models.FullDecRefModelV2.Collator import FullDecRefModelV2Collator
from models.reference_models.FullDecRefModelV2.Preprocess import FullDecRefModelV2Preprocess
from models.reference_models.FullDecRefModelV2.manager import FullDecRefModelV2Manager
from models.reference_models.FullDecRefModelV2.model import FullDecRefModelV2
from models.reference_models.FullDecUtilityModel.Collator import FullDecUtilityModelCollator
from models.reference_models.FullDecUtilityModel.Preprocess import FullDecUtilityModelPreprocess
from models.reference_models.FullDecUtilityModel.manager import FullDecUtilityBaseManager
from models.reference_models.FullDecUtilityModel.model import FullDecUtilityBaseModel


class FullDecRefModelV2Info:

    manager = FullDecRefModelV2Manager
    model = FullDecRefModelV2
    preprocess = FullDecRefModelV2Preprocess
    collate = FullDecRefModelV2Collator