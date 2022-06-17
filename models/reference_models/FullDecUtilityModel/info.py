

from models.reference_models.FullDecUtilityModel.Collator import FullDecUtilityModelCollator
from models.reference_models.FullDecUtilityModel.Preprocess import FullDecUtilityModelPreprocess
from models.reference_models.FullDecUtilityModel.manager import FullDecUtilityModelManager
from models.reference_models.FullDecUtilityModel.model import FullDecUtilityModel


class FullDecUtilityModelInfo:

    manager = FullDecUtilityModelManager
    model = FullDecUtilityModel
    preprocess = FullDecUtilityModelPreprocess
    collate = FullDecUtilityModelCollator