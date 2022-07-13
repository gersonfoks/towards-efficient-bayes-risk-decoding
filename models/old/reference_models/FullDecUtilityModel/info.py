

from models.old.reference_models.FullDecUtilityModel.Collator import FullDecUtilityModelCollator
from models.old.reference_models.FullDecUtilityModel.Preprocess import FullDecUtilityModelPreprocess
from models.old.reference_models.FullDecUtilityModel.manager import FullDecUtilityBaseManager
from models.old.reference_models.FullDecUtilityModel.model import FullDecUtilityBaseModel


class FullDecUtilityModelInfo:

    manager = FullDecUtilityBaseManager
    model = FullDecUtilityBaseModel
    preprocess = FullDecUtilityModelPreprocess
    collate = FullDecUtilityModelCollator