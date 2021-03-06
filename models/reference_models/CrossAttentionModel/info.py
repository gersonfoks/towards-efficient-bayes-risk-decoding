from models.reference_models.BasicCrossAttentionModel.Collator import BasicCrossAttentionModelCollator
from models.reference_models.BasicCrossAttentionModel.Preprocess import BasicCrossAttentionModelPreprocess
from models.reference_models.BasicCrossAttentionModel.manager import BasicCrossAttentionBaseManager
from models.reference_models.BasicCrossAttentionModel.model import BasicCrossAttentionModel
from models.reference_models.CrossAttentionModel.Collator import CrossAttentionModelCollator
from models.reference_models.CrossAttentionModel.Preprocess import CrossAttentionModelPreprocess
from models.reference_models.CrossAttentionModel.manager import CrossAttentionModelManager
from models.reference_models.CrossAttentionModel.model import CrossAttentionModel


class CrossAttentionModelInfo:


    manager = CrossAttentionModelManager
    model = CrossAttentionModel
    preprocess = CrossAttentionModelPreprocess
    collator = CrossAttentionModelCollator