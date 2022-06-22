from models.reference_models.LastHiddenStateRefModel.Collator import LastHiddenStateRefModelCollator
from models.reference_models.LastHiddenStateRefModel.Preprocess import LastHiddenStateRefModelPreprocessor
from models.reference_models.LastHiddenStateRefModel.manager import LastHiddenStateRefBaseManager
from models.reference_models.LastHiddenStateRefModel.model import LastHiddenStateRefModel


class LastHiddenStateRefModelInfo:


    manager = LastHiddenStateRefBaseManager
    model = LastHiddenStateRefModel
    preprocess = LastHiddenStateRefModelPreprocessor
    collator = LastHiddenStateRefModelCollator