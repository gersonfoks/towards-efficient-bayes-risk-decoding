from models.reference_models.CrossAttentionModel.Collator import CrossAttentionModelCollator
from models.reference_models.CrossAttentionModel.Preprocess import CrossAttentionModelPreprocess


from models.reference_models.CrossAttentionModelV4.manager import CrossAttentionModelV4Manager
from models.reference_models.CrossAttentionModelV4.model import CrossAttentionModelV4


class CrossAttentionModelV4Info:
    manager = CrossAttentionModelV4Manager
    model = CrossAttentionModelV4
    preprocess = CrossAttentionModelPreprocess
    collator = CrossAttentionModelCollator
