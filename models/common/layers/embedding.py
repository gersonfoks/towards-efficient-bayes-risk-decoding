import torch
from torch import nn


class EmbbedingForPackedSequenceLayer(torch.nn.Module):
    """
    Lstm embedding layer that handles packed sequences
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding = torch.nn.Embedding(input_dim, output_dim)

    def forward(self, packed_sequence):
        return self.element_wise_apply(self.embedding, packed_sequence)

    def element_wise_apply(self, fn, packed_sequence):
        # from: https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-are-going-to-use-word-embedding-and-bilstm/28184/3
        """applies a pointwise function fn to each element in packed_sequence"""

        return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes,
                                                 sorted_indices=packed_sequence.sorted_indices,
                                                 unsorted_indices=packed_sequence.unsorted_indices)


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
        return None


class HiddenStateEmbedding(nn.Module):

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
        # print(list(nmt_out["decoder_hidden_states"]))
        return nmt_out["decoder_hidden_states"], attention_mask_decoder

    def parameters(self, recurse: bool = True):
        return []


class EncDecLastStateEmbedding(nn.Module):

    def __init__(self, nmt_model, padding_id=58100):
        super().__init__()
        self.nmt_model = nmt_model
        self.padding_id = padding_id

    @torch.no_grad()
    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None):
        with torch.no_grad():
            nmt_out = self.nmt_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                             decoder_input_ids=decoder_input_ids, output_hidden_states=True,
                                             output_attentions=True)
        attention_mask_decoder = (self.padding_id != labels).long()
        return nmt_out["encoder_hidden_states"][-1], attention_mask, nmt_out["decoder_hidden_states"][
            -1], attention_mask_decoder,

    def parameters(self, recurse: bool = True):
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