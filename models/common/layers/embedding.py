import torch
from torch import nn

from models.common.layers.pooling import GlobalMeanPooling


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
        self.embedding = nn.Linear(7,
                                   embedding_size)  # We use 7 statistics namely the log probability, the log entropy and finally the top 5 log probability of the tokens

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None):
        self.nmt_model.eval()

        with torch.no_grad():
            nmt_out = self.nmt_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                             decoder_input_ids=decoder_input_ids, output_hidden_states=True,
                                             output_attentions=True)

            attention_mask_decoder = (self.padding_id != labels).long()

            statistics = logits_to_statistics(nmt_out["logits"], labels)

        embedding = self.embedding(statistics)

        return embedding, attention_mask_decoder

    def parameters(self, recurse: bool = True):
        return self.embedding.parameters()


class FullDecEmbedding(nn.Module):

    def __init__(self, nmt_model, embedding_size, padding_id=-100):
        super().__init__()
        self.nmt_model = nmt_model

        self.padding_id = padding_id
        if embedding_size != None:
            self.embedding = nn.Linear(7,
                                       embedding_size)  # We use two statistics namely the probability and the log entropy
        else:
            self.embedding = None

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None):
        self.nmt_model.eval()
        with torch.no_grad():
            nmt_out = self.nmt_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                             decoder_input_ids=decoder_input_ids, output_hidden_states=True,
                                             output_attentions=True)

            attention_mask_decoder = (self.padding_id != labels).long()

            if self.embedding != None:
                statistics = logits_to_statistics(nmt_out["logits"], labels)
        if self.embedding != None:
            embedding = self.embedding(statistics)
        else:
            embedding = None

        # print(list(nmt_out["decoder_hidden_states"]))
        return nmt_out["decoder_hidden_states"], embedding, attention_mask_decoder

    def parameters(self, recurse: bool = True):
        if self.embedding:
            return self.embedding.parameters()
        else:
            return []


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

    top_5 = torch.topk(probs_all_tokens, 5, dim=-1, ).values

    statistics = torch.concat([probs, entropy, top_5], dim=-1)

    return statistics


class UnigramCountEmbedding(nn.Module):

    def __init__(self, embedding, padding_id=-100):
        super().__init__()
        self.embedding = embedding.to("cuda")
        self.padding_id = padding_id
        self.mean_pooling = GlobalMeanPooling()

    @torch.no_grad()
    def forward(self, input_ids):
        with torch.no_grad():
            embedding = self.embedding(input_ids)
            padding = input_ids == self.padding_id

            mean_embedding = self.mean_pooling(embedding, padding)

        return mean_embedding

    def parameters(self, recurse: bool = True):
        return []
