import torch
from torch import nn
from models.models import Transformer

class transformer_model(nn.Module):
    
    def __init__(self, config):
        super(transformer_model, self).__init__()
        self.config = config
        self.transformer = Transformer(
            n_src_vocab=config["n_src_vocab"], d_word_vec=config["d_model"], n_layers=config["n_layers"], n_head=config["n_head"], d_k=config["d_k"], d_v=config["d_v"],
            d_model=config["d_model"], d_inner=config["d_inner"], dropout=config["dropout"], n_position=config["n_position"], n_trg_vocab=config["n_trg_vocab"], src_pad_idx=config["pad_idx"], trg_pad_idx=config["pad_idx"])
        self.linear = nn.Linear(config["d_model"], config["n_trg_vocab"])
        
    def forward(self, src_seq, trg_seq):
        preds = self.transformer(src_seq, trg_seq)
        output = self.linear(preds)
        return output
    
    def generate(self, src_seq, config, max_len=20):
        generated_seq = torch.zeros(1, max_len).long().to(src_seq.device)

        generated_seq[0, 0] = config["en_word_to_idx"]["<sos>"]

        for t in range(1, max_len):
            preds = self.transformer(src_seq, generated_seq[:, :t])
            predicted_token = torch.argmax(preds[0, -1, :])
            generated_seq[0, t] = predicted_token

        return generated_seq