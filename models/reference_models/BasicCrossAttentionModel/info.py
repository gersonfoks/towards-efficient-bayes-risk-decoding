from models.reference_models.BasicCrossAttentionModel.Collator import BasicCrossAttentionModelCollator
from models.reference_models.BasicCrossAttentionModel.Preprocess import BasicCrossAttentionModelPreprocess
from models.reference_models.BasicCrossAttentionModel.manager import BasicCrossAttentionModelManager
from models.reference_models.BasicCrossAttentionModel.model import BasicCrossAttentionModel



class BasicCrossAttentionModelInfo:


    manager = BasicCrossAttentionModelManager
    model = BasicCrossAttentionModel
    preprocess = BasicCrossAttentionModelPreprocess
    collator = BasicCrossAttentionModelCollator