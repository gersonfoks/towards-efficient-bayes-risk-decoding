from models.old.reference_models.FullDecRefModelV2.Collator import FullDecRefModelV2Collator
from models.old.reference_models.FullDecRefModelV2.Preprocess import FullDecRefModelV2Preprocess
from models.old.reference_models.FullDecRefModelV3.manager import FullDecRefModelV3Manager
from models.old.reference_models.FullDecRefModelV3.model import FullDecRefModelV3


class FullDecRefModelV3Info:

    manager = FullDecRefModelV3Manager
    model = FullDecRefModelV3
    preprocess = FullDecRefModelV2Preprocess
    collate = FullDecRefModelV2Collator