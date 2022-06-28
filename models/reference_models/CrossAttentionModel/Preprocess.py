from datasets import Dataset

from models.common.preprocessing import add_hypothesis_ids, explode_dataset, get_prob_entropy_lookup_table
from utilities.wrappers.NmtWrapper import NMTWrapper


class CrossAttentionModelPreprocess:
    '''
    Splits the dataset hypotheses column and takes the average of the unigram f1 scores
    '''

    def __init__(self, nmt_model, tokenizer, max_seq_length=75, table_location='./'):
        self.nmt_model = nmt_model
        self.nmt_model_wrapped = NMTWrapper(self.nmt_model, tokenizer)
        self.nmt_model.eval()
        self.nmt_model = self.nmt_model.to("cuda")
        self.tokenizer = tokenizer

        self.max_seq_length = max_seq_length

        self.batch_size = 32

        self.table_location = table_location

    def __call__(self, data):
        data = data.reset_index()
        data["references"] = data["hypotheses"]
        data["reference_counts"] = data["count"]

        data = add_hypothesis_ids(data)

        data["reference_ids"] = data["hypotheses_ids"]

        data = explode_dataset(data)

        dataset = Dataset.from_pandas(data)

        prob_entropy_lookup_table = get_prob_entropy_lookup_table(dataset, self.nmt_model_wrapped,
                                                                  location=self.table_location)

        return dataset, prob_entropy_lookup_table
