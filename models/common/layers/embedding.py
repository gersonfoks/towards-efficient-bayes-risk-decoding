import torch
from torch import nn


class LastStateEmbedding(nn.Module):

    def __init__(self, nmt_model, padding_id=-100):
        super().__init__()
        self.nmt_model = nmt_model
        self.padding_id = padding_id

    @torch.no_grad()
    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None):
        self.nmt_model.eval()
        with torch.no_grad():
            nmt_out = self.nmt_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                             decoder_input_ids=decoder_input_ids, output_hidden_states=True,
                                             output_attentions=True)
        attention_mask_decoder = (self.padding_id != labels).long()
        return nmt_out["decoder_hidden_states"][-1], attention_mask_decoder

    def parameters(self, recurse: bool = True):
        return []


class TokenStatisticsEmbedding(nn.Module):

    def __init__(self, nmt_model, embedding_size, padding_id=-100):
        super().__init__()
        self.nmt_model = nmt_model
        self.padding_id = padding_id
        self.embedding = nn.Linear(2,
                                   embedding_size)  # We use two statistics namely the probability and the log entropy

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None):
        self.nmt_model.eval()

        with torch.no_grad():
            nmt_out = self.nmt_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                             decoder_input_ids=decoder_input_ids, output_hidden_states=True,
                                             output_attentions=True)

            attention_mask_decoder = (self.padding_id != labels).long()

            statistics = logits_to_statistics(nmt_out["logits"], attention_mask_decoder)

        embedding = self.embedding(statistics)

        return embedding, attention_mask_decoder

    def parameters(self, recurse: bool = True):
        return self.embedding.parameters()


class FullDecEmbedding(nn.Module):

    def __init__(self, nmt_model, embedding_size, padding_id=-100):
        super().__init__()
        self.nmt_model = nmt_model

        self.padding_id = padding_id

        self.embedding = nn.Linear(2,
                                   embedding_size)  # We use two statistics namely the probability and the log entropy

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None):
        self.nmt_model.eval()
        with torch.no_grad():
            nmt_out = self.nmt_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                             decoder_input_ids=decoder_input_ids, output_hidden_states=True,
                                             output_attentions=True)

            attention_mask_decoder = (self.padding_id != labels).long()

            statistics = logits_to_statistics(nmt_out["logits"], attention_mask_decoder)
        embedding = self.embedding(statistics)

        # print(list(nmt_out["decoder_hidden_states"]))
        return nmt_out["decoder_hidden_states"], embedding, attention_mask_decoder

    def parameters(self, recurse: bool = True):
        return self.embedding.parameters()



class CometEmbedding(nn.Module):

    def __init__(self, comet_model, feed_forward):
        super().__init__()
        self.comet_model = comet_model

        self.feed_forward = feed_forward

        self.device = 'cuda'

    def forward(self, source, hypothesis, references):
        comet_embeddings = [

        ]

        with torch.no_grad():
            self.comet_model.eval()
            src_inputs = self.comet_model.encoder.prepare_sample(source).to(self.device)
            hyp_inputs = self.comet_model.encoder.prepare_sample(hypothesis).to(self.device)

            src_sent_embed = self.comet_model.get_sentence_embedding(**src_inputs)
            hyp_sent_embed = self.comet_model.get_sentence_embedding(**hyp_inputs)

            scores = []

            for refs in references:
                ref_inputs = self.comet_model.encoder.prepare_sample(refs.tolist()).to(self.device)

                ref_sent_embed = self.comet_model.get_sentence_embedding(**ref_inputs)

                diff_ref = torch.abs(hyp_sent_embed - ref_sent_embed)
                diff_src = torch.abs(hyp_sent_embed - src_sent_embed)

                prod_ref = hyp_sent_embed * ref_sent_embed
                prod_src = hyp_sent_embed * src_sent_embed

                embedded_sequences = torch.cat(
                    (hyp_sent_embed, ref_sent_embed, prod_ref, diff_ref, prod_src, diff_src),
                    dim=1, )

                # Forward through feed forward (project down)
                comet_embeddings.append(embedded_sequences)

                scores.append(self.comet_model.estimator(embedded_sequences))

        # Get the final embeddings (project down)
        embeddings = []
        for x in comet_embeddings:
            emb = self.feed_forward(x)
            embeddings.append(emb)
        # Stack to get a sequence
        embeddings = torch.stack(embeddings, dim=1)
        scores = torch.concat(scores, dim=-1)

        scores = torch.mean(scores, dim=-1).unsqueeze(dim=-1)

        return embeddings, scores

    def parameters(self, recurse: bool = True):
        return self.feed_forward.parameters()


def logits_to_statistics(logits, labels):
    hyp_input_ids = (labels * (labels != -100)).long()  # Get the indices
    ids = hyp_input_ids.unsqueeze(dim=-1)

    log_softmax = torch.nn.LogSoftmax(dim=-1)

    log_probs_all_tokens = log_softmax(logits)
    probs_all_tokens = torch.exp(log_probs_all_tokens)
    entropy = - torch.sum(log_probs_all_tokens * probs_all_tokens, dim=-1).unsqueeze(dim=-1)

    probs = probs_all_tokens.gather(-1, ids)

    statistics = torch.concat([probs, entropy], dim=-1)

    return statistics
