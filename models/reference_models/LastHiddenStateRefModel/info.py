from models.reference_models.LastHiddenStateRefModel.Collator import LastHiddenStateRefModelCollator
from models.reference_models.LastHiddenStateRefModel.Preprocess import LastHiddenStateRefModelPreprocessor
from models.reference_models.LastHiddenStateRefModel.manager import LastHiddenStateRefModelManager
from models.reference_models.LastHiddenStateRefModel.model import LastHiddenStateRefModel


class LastHiddenStateRefModelInfo:


    manager = LastHiddenStateRefModelManager
    model = LastHiddenStateRefModel
    preprocess = LastHiddenStateRefModelPreprocessor
    collator = LastHiddenStateRefModelCollator