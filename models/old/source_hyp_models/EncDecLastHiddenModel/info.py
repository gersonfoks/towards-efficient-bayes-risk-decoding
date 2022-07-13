

# This class contains all the information for the training and hyperparam search
# It aids reusability.
from models.old.source_hyp_models import EncDecLastHiddenModel
from models.old.source_hyp_models.EncDecLastHiddenModel.Collator import EncDecLastHiddenCollator
from models.old.source_hyp_models.EncDecLastHiddenModel.Preprocess import EncDecLastHiddenModelPreprocess
from models.old.source_hyp_models.EncDecLastHiddenModel.manager import EncDecLastHiddenBaseManager


class EncDecLastHiddenModelInfo:

    manager = EncDecLastHiddenBaseManager
    model = EncDecLastHiddenModel
    preprocess = EncDecLastHiddenModelPreprocess
    collate = EncDecLastHiddenCollator