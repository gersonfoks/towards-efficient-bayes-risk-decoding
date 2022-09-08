import torch
from tqdm import tqdm

from utilities.misc import batch


class CometWrapper:

    def __init__(self, cometModel, device='cuda'):
        super().__init__()

        self.model = cometModel

        self.device = device

    @torch.no_grad()
    def to_embedding(self, strings):
        inputs = self.model.encoder.prepare_sample(strings).to(self.device)

        return self.model.get_sentence_embedding(**inputs)

    @torch.no_grad()
    def predict(self, samples):
        prepared_samples = self.model.prepare_for_inference(samples, )
        inputs = {k: v.to(self.device) for k, v in prepared_samples.items()}
        return self.model.forward(**inputs)

    @torch.no_grad()
    def batch_predict(self, sources, hypotheses, targets, batch_size=64):
        to_predict = []
        result = []

        for source, hypothesis, target in tqdm(zip(sources, hypotheses, targets), total=len(sources)):

            if len(to_predict) < batch_size:
                to_predict.append(
                    {
                        "src": source,
                        "mt": hypothesis,
                        "ref": target
                    }
                )
            else:
                prediction = self.predict(to_predict)["score"].cpu().numpy().flatten().tolist()
                result += prediction
                to_predict = []

        # Last batch
        prediction = self.predict(to_predict)["score"].cpu().numpy().flatten().tolist()
        result += prediction
        return result

    @torch.no_grad()
    def fast_predict(self, source, hypothesis, refs):
        src_inputs = self.model.encoder.prepare_sample([source]).to(self.device)
        hyp_inputs = self.model.encoder.prepare_sample(hypothesis).to(self.device)
        ref_inputs = self.model.encoder.prepare_sample(refs).to(self.device)

        n_refs = len(refs)

        src_sent_embed = self.model.get_sentence_embedding(**src_inputs)
        hyp_sent_embed = self.model.get_sentence_embedding(**hyp_inputs)
        ref_sent_embed = self.model.get_sentence_embedding(**ref_inputs)

        src_sent_embed = src_sent_embed.repeat(n_refs, 1)

        scores = []
        for h in hyp_sent_embed:
            h = h.unsqueeze(dim=0).repeat(n_refs, 1)
            diff_ref = torch.abs(h - ref_sent_embed)
            diff_src = torch.abs(h - src_sent_embed)

            prod_ref = h * ref_sent_embed
            prod_src = h * src_sent_embed

            embedded_sequences = torch.cat(
                (h, ref_sent_embed, prod_ref, diff_ref, prod_src, diff_src),
                dim=1, )

            scores.append(self.model.estimator(embedded_sequences).cpu().numpy().flatten())

        return scores

    @torch.no_grad()
    def fast_predict_batched(self, source, hypothesis, references, hyp_batch_size=100, ref_batch_size=100):

        # We need to keep track of the scores for each hypotheses
        scores = [[] for i in range(len(hypothesis))]
        id_map = {h: i for i, h in enumerate(hypothesis)}

        src_inputs = self.model.encoder.prepare_sample([source]).to(self.device)
        src_sent_embed = self.model.get_sentence_embedding(**src_inputs)
        for refs in batch(references, n=ref_batch_size):

            ref_inputs = self.model.encoder.prepare_sample(refs).to(self.device)

            n_refs = len(refs)

            src_sent_embed_repeated = src_sent_embed.repeat(len(refs), 1)

            ref_sent_embed = self.model.get_sentence_embedding(**ref_inputs)

            for hyp in batch(hypothesis, n=hyp_batch_size):

                hyp_inputs = self.model.encoder.prepare_sample(hyp).to(self.device)
                hyp_sent_embed = self.model.get_sentence_embedding(**hyp_inputs)

                for h_sent_embed, h in zip(hyp_sent_embed, hyp):
                    i = id_map[h]

                    h_sent_embed = h_sent_embed.unsqueeze(dim=0).repeat(n_refs, 1)
                    diff_ref = torch.abs(h_sent_embed - ref_sent_embed)
                    diff_src = torch.abs(h_sent_embed - src_sent_embed_repeated)

                    prod_ref = h_sent_embed * ref_sent_embed
                    prod_src = h_sent_embed * src_sent_embed_repeated

                    embedded_sequences = torch.cat(
                        (h_sent_embed, ref_sent_embed, prod_ref, diff_ref, prod_src, diff_src),
                        dim=1, )

                    scores[i] += list(self.model.estimator(embedded_sequences).cpu().numpy().flatten())
        return scores


