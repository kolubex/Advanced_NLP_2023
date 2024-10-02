import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from utils.train_eval_utils import *

class SquadQADataset(Dataset):
    def __init__(self, dataframe, config, tokenizer, model_name="gpt2"):
        # self.dataframe = pd.read_csv(dataframe).drop(columns=["id"])
        #random sample 1000 rows
        # self.dataframe = self.dataframe.sample(frac=0.1, random_state=42).reset_index(drop=True)
        #remove rows with any nan values
        self.dataframe = dataframe.dropna()
        self.config = config
        self.model_name = model_name
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        input_query = "[Context]"+self.dataframe.iloc[idx]["Context"]+ "[Question]" + self.dataframe.iloc[idx]["Question"]
        article = self.tokenizer(input_query, padding='max_length', return_tensors="pt", truncation=True)
        label = self.tokenizer(self.dataframe.iloc[idx]["Answer"], padding='max_length', return_tensors="pt", truncation=True)
        label_text = self.dataframe.iloc[idx]["Answer"]
        return article, label, label_text
    
    def __len__(self):
        return len(self.dataframe)
    
    