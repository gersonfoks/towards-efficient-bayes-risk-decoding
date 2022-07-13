from models.old.reference_models.CrossAttentionModel.Collator import CrossAttentionModelCollator
from models.old.reference_models.CrossAttentionModel.Preprocess import CrossAttentionModelPreprocess
from models.old.reference_models.CrossAttentionModel.manager import CrossAttentionModelManager
from models.old.reference_models.CrossAttentionModel.model import CrossAttentionModel


class CrossAttentionModelInfo:


    manager = CrossAttentionModelManager
    model = CrossAttentionModel
    preprocess = CrossAttentionModelPreprocess
    collator = CrossAttentionModelCollator