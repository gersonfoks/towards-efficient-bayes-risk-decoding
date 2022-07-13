
from models.old.reference_models.FullDecRefModelV4.Collator import FullDecRefModelV4Collator
from models.old.reference_models.FullDecRefModelV4.Preprocess import FullDecRefModelV4Preprocess
from models.old.reference_models.FullDecRefModelV4.manager import FullDecRefModelV4Manager
from models.old.reference_models.FullDecRefModelV4.model import FullDecRefModelV4


class FullDecRefModelV4Info:

    manager = FullDecRefModelV4Manager
    model = FullDecRefModelV4
    preprocess = FullDecRefModelV4Preprocess
    collate = FullDecRefModelV4Collator