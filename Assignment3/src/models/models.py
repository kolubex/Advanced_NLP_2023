''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from .layers import EncoderLayer, DecoderLayer


__author__ = "kolubex"


def get_pad_mask(seq, pad_idx):
    '''
        Description: Create a padding mask to mask out padding positions.
        
        Args:
        - seq (Tensor): The input sequence.
        - pad_idx (int): The padding index.

        Returns:
        - mask (Tensor): The padding mask.
    '''
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):
    '''
        Description: This class implements positional encoding for the input sequences.

        Args:
        - d_hid (int): The dimension of the positional encoding.
        - n_position (int): The maximum sequence length.

        Returns:
        - x (Tensor): The input sequence with positional encoding added.
    '''
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    '''
        Description: This class defines the encoder part of the Transformer model.

        Args:
        - n_src_vocab (int): The size of the source vocabulary.
        - d_word_vec (int): The dimension of the word embeddings.
        - n_layers (int): The number of encoder layers.
        - n_head (int): The number of attention heads.
        - d_k (int): The dimension of the keys in the attention mechanism.
        - d_v (int): The dimension of the values in the attention mechanism.
        - d_model (int): The model's dimension.
        - d_inner (int): The dimension of the inner layer in the position-wise feed-forward sub-layer.
        - pad_idx (int): The padding index.
        - dropout (float): The dropout probability for regularization (default is 0.1).
        - n_position (int): The maximum sequence length.
        - scale_emb (bool): Whether to scale the embeddings (default is False).

        Returns:
        - enc_output (Tensor): The output of the encoder.
    '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    '''
        Description: This class defines the decoder part of the Transformer model.

        Args:
        - n_trg_vocab (int): The size of the target vocabulary.
        - d_word_vec (int): The dimension of the word embeddings.
        - n_layers (int): The number of decoder layers.
        - n_head (int): The number of attention heads.
        - d_k (int): The dimension of the keys in the attention mechanism.
        - d_v (int): The dimension of the values in the attention mechanism.
        - d_model (int): The model's dimension.
        - d_inner (int): The dimension of the inner layer in the position-wise feed-forward sub-layer.
        - pad_idx (int): The padding index.
        - n_position (int): The maximum sequence length.
        - dropout (float): The dropout probability for regularization (default is 0.1).
        - scale_emb (bool): Whether to scale the embeddings (default is False).

        Returns:
        - dec_output (Tensor): The output of the decoder.
    '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    '''
        Description: This class defines the Transformer model, which consists of an encoder and a decoder.

        Args:
        - n_src_vocab (int): The size of the source vocabulary.
        - n_trg_vocab (int): The size of the target vocabulary.
        - src_pad_idx (int): The padding index for the source.
        - trg_pad_idx (int): The padding index for the target.
        - d_word_vec (int): The dimension of the word embeddings.
        - d_model (int): The model's dimension.
        - d_inner (int): The dimension of the inner layer in the position-wise feed-forward sub-layer.
        - n_layers (int): The number of encoder and decoder layers.
        - n_head (int): The number of attention heads.
        - d_k (int): The dimension of the keys in the attention mechanism.
        - d_v (int): The dimension of the values in the attention mechanism.
        - dropout (float): The dropout probability for regularization (default is 0.1).
        - n_position (int): The maximum sequence length.
        - trg_emb_prj_weight_sharing (bool): Whether to share weights between target word embeddings and the last dense layer.
        - emb_src_trg_weight_sharing (bool): Whether to share weights between source and target word embeddings.
        - scale_emb_or_prj (str): Scaling option for embeddings or projection ('emb', 'prj', 'none').

        Returns:
        - dec_output (Tensor): The output of the Transformer model.
    '''

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        # self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        # if trg_emb_prj_weight_sharing:
        #     # Share the weight between target word embedding & last dense layer
        #     self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq, trg_seq):

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        # seq_logit = self.trg_word_prj(dec_output)
        # if self.scale_prj:
        #     seq_logit *= self.d_model ** -0.5
        return dec_output
