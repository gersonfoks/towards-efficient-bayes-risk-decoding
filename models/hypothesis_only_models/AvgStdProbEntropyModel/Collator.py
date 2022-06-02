import torch


class AvgStdProbEntropyModelCollator:
    '''
    Tokenizes the hypotheses and put them into a packed sequence (as we work with lstms)
    '''

    def __init__(self, device="cuda"):
        self.device = device

    def __call__(self, batch):
        hypotheses = [b["hypotheses"] for b in batch]
        sources = [b["source"] for b in batch]

        score = torch.tensor([b["score"] for b in batch])

        # We get the lenghts of the sequences (used for packing later)

        # Setup the tokenizer for targets

        mean_log_prob = torch.tensor([b["mean_log_prob"] for b in batch]).unsqueeze(-1)
        std_log_prob = torch.tensor([b["std_log_prob"] for b in batch]).unsqueeze(-1)
        mean_entropy = torch.tensor([b["mean_entropy"] for b in batch]).unsqueeze(-1)
        std_entropy = torch.tensor([b["std_entropy"] for b in batch]).unsqueeze(-1)

        features = {
            "mean_log_prob": mean_log_prob,
            "std_log_prob": std_log_prob,
            "mean_entropy": mean_entropy,
            "std_entropy": std_entropy,

        }
        return sources, hypotheses, features, score
