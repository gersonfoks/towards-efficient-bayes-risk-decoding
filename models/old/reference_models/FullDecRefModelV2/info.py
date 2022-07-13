from models.old.reference_models.FullDecRefModelV2.Collator import FullDecRefModelV2Collator
from models.old.reference_models.FullDecRefModelV2.Preprocess import FullDecRefModelV2Preprocess
from models.old.reference_models.FullDecRefModelV2.manager import FullDecRefModelV2Manager
from models.old.reference_models.FullDecRefModelV2.model import FullDecRefModelV2


class FullDecRefModelV2Info:

    manager = FullDecRefModelV2Manager
    model = FullDecRefModelV2
    preprocess = FullDecRefModelV2Preprocess
    collate = FullDecRefModelV2Collator