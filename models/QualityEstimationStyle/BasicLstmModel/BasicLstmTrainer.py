

from models.Base.BaseTrainer import BaseTrainer
from models.QualityEstimationStyle.BasicLstmModel.BasicLstmModelManager import BasicLstmModelManager


class BasicLstmModelTrainer(BaseTrainer):

    def __init__(self, config, smoke_test=False):
        super().__init__(config, smoke_test)

        self.manager_class = BasicLstmModelManager







