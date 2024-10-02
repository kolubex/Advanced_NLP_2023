import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.train_eval_utils import *
from transformers import AutoTokenizer

class softprompting(nn.Module):
    def __init__(self, wte, tokenizer, prompt):
        """
        wte: word token embeddings.
        n_tokens: number of tokens to be kept for the soft prompt (trainable).
        """
        super(softprompting, self).__init__()
        self.tokenizer = tokenizer
        self.wte = wte
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding())
    
    
    def initialize_embedding(self):
        tokenized_prompt = self.tokenizer(self.prompt)["input_ids"]
        self.len_soft_prompt = len(tokenized_prompt)
        prompt_embedding = self.wte.weight[tokenized_prompt]
                
        return prompt_embedding

        
    def forward(self, tokens):
        """
        tokens: input tokens from the model, prior to soft prompt learnable embeddings replacement.
        """
        input_embed = self.wte(tokens[:, self.len_soft_prompt:])
        learned_embedding = self.learned_embedding.repeat(input_embed.size(0), 1, 1)
        return torch.cat((learned_embedding, input_embed), dim=1)