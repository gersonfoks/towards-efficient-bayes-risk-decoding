from models.reference_models.CrossAttentionModel.Collator import CrossAttentionModelCollator
from models.reference_models.CrossAttentionModel.Preprocess import CrossAttentionModelPreprocess

from models.reference_models.CrossAttentionModelV3.manager import CrossAttentionModelV3Manager
from models.reference_models.CrossAttentionModelV3.model import CrossAttentionModelV3


class CrossAttentionModelV3Info:
    manager = CrossAttentionModelV3Manager
    model = CrossAttentionModelV3
    preprocess = CrossAttentionModelPreprocess
    collator = CrossAttentionModelCollator
