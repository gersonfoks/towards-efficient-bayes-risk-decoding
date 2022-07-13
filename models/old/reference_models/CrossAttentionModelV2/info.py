from models.old.reference_models.CrossAttentionModel.Collator import CrossAttentionModelCollator
from models.old.reference_models.CrossAttentionModel.Preprocess import CrossAttentionModelPreprocess

from models.old.reference_models.CrossAttentionModelV2.manager import CrossAttentionModelV2Manager
from models.old.reference_models.CrossAttentionModelV2.model import CrossAttentionModelV2


class CrossAttentionModelV2Info:


    manager = CrossAttentionModelV2Manager
    model = CrossAttentionModelV2
    preprocess = CrossAttentionModelPreprocess
    collator = CrossAttentionModelCollator