import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "kolubex"

class ScaledDotProductAttention(nn.Module):
    ''' 
        Scaled Dot-Product Attention 
        Args:
            - temperature (float): The temperature parameter.
            - attn_dropout (float): The dropout probability for regularization (default is 0.1).

        Returns:
            - output (Tensor): The output of the scaled dot-product attention.
            - attn (Tensor): The attention scores.
    '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
