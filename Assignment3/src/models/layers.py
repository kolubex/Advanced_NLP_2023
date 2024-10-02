''' Define the Layers '''
import torch.nn as nn
import torch
from .sublayers import MultiHeadAttention, PositionwiseFeedForward


__author__ = "kolubex"


class EncoderLayer(nn.Module):
    '''
        Description: This class defines an encoder layer, which composes of two sub-layers, multi-head self-attention and position-wise feed-forward.
        
        Args:
        - d_model (int): The dimension of the model.
        - d_inner (int): The dimension of the inner layer in the position-wise feed-forward sub-layer.
        - n_head (int): The number of attention heads in the multi-head self-attention sub-layer.
        - d_k (int): The dimension of the keys in the attention mechanism.
        - d_v (int): The dimension of the values in the attention mechanism.
        - dropout (float): The dropout probability for regularization (default is 0.1).
        
        Returns:
        - enc_output (Tensor): The output of the encoder layer.
        - enc_slf_attn: The self-attention scores for the encoder layer.
    '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        '''
            Args:
            - enc_input (Tensor): The input to the encoder layer.
            - slf_attn_mask (Tensor): The self-attention mask for masking out padding positions (default is None).
            
            Returns:
            - enc_output (Tensor): The output of the encoder layer.
            - enc_slf_attn (Tensor): The self-attention scores for the encoder layer.
        '''
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    '''
        Description: This class defines a decoder layer, which composes of three sub-layers, multi-head self-attention, multi-head encoder-decoder attention, and position-wise feed-forward.
        
        Args:
        - d_model (int): The dimension of the model.
        - d_inner (int): The dimension of the inner layer in the position-wise feed-forward sub-layer.
        - n_head (int): The number of attention heads in the multi-head self-attention and encoder-decoder attention sub-layers.
        - d_k (int): The dimension of the keys in the attention mechanisms.
        - d_v (int): The dimension of the values in the attention mechanisms.
        - dropout (float): The dropout probability for regularization (default is 0.1).
        
        Returns:
        - dec_output (Tensor): The output of the decoder layer.
        - dec_slf_attn (Tensor): The self-attention scores for the decoder layer.
        - dec_enc_attn (Tensor): The encoder-decoder attention scores for the decoder layer.
    '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        '''
            Args:
            - dec_input (Tensor): The input to the decoder layer.
            - enc_output (Tensor): The output of the encoder for encoder-decoder attention.
            - slf_attn_mask (Tensor): The self-attention mask for masking out padding positions (default is None).
            - dec_enc_attn_mask (Tensor): The encoder-decoder attention mask (default is None).
            
            Returns:
            - dec_output (Tensor): The output of the decoder layer.
            - dec_slf_attn (Tensor): The self-attention scores for the decoder layer.
            - dec_enc_attn (Tensor): The encoder-decoder attention scores for the decoder layer.
        '''
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
