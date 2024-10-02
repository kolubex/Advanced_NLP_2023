import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from utils.train_eval_utils import *

class EuroPalDataset(Dataset):
    def __init__(self, dataframe, config, tokenizer, model_name="gpt2"):
        #remove rows with any nan values
        self.dataframe = dataframe.dropna()
        self.config = config
        self.model_name = model_name
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        article = self.tokenizer(self.dataframe.iloc[idx]["English"], padding='max_length', return_tensors="pt", truncation=True)
        label = self.tokenizer(self.dataframe.iloc[idx]["German"], padding='max_length', return_tensors="pt", truncation=True)
        label_text = self.dataframe.iloc[idx]["German"]
        return article, label, label_text
    
    def __len__(self):
        return len(self.dataframe)
        