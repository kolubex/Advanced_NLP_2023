import torch
import torch.nn as nn
import os
import yaml
import argparse
from utils.train_eval_utils import *
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import wandb
from dataloader.cnn_dataset import CNN_dataset
from dataloader.squad_dataset import SquadQADataset
from dataloader.europal_dataset import EuroPalDataset
from utils.softprompting import softprompting
from torch.optim import Adam
import transformers
from transformers import logging as hf_logging
import pandas as pd
# import omegaconf
from omegaconf import OmegaConf

def trigger_training(config, model_name, device):
    task = config["task"]
    train_df = config[task]["paths"]["train_df"]
    val_df = config[task]["paths"]["val_df"]
    config["wandb_run_name"] = config[task]["wandb_run_name"]
    config["model_save_path"] = config[task]["model_save_path"]
    config["prompt"] = config[task]["soft_prompt"]
    if config["task"] == "summarization":
        train_df = pd.read_csv(train_df).drop(columns=["id"]).sample(frac=0.2, random_state=42).reset_index(drop=True)
        val_df = pd.read_csv(val_df).drop(columns=["id"]).sample(frac=0.2, random_state=42).reset_index(drop=True)
    if config["task"] == "squadqa":
        train_df = pd.read_csv(train_df).sample(frac=0.2, random_state=42).reset_index(drop=True)
        val_df = pd.read_csv(val_df).sample(frac=0.2, random_state=42).reset_index(drop=True)
    if config["task"] == "europal":
        train_df = pd.read_csv(train_df)
        val_df = pd.read_csv(val_df)
        train_df = train_df.sample(frac=0.4, random_state=42).reset_index(drop=True)
        val_df = val_df.sample(frac=0.4, random_state=42).reset_index(drop=True)


    model = GPT2LMHeadModel.from_pretrained(config["model_name"])
    tokenizer = GPT2TokenizerFast.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token


    if config["task"] == "summarization":
        train_dataset = CNN_dataset(dataframe = train_df,
                                      config = config,
                                      tokenizer = tokenizer)
        
        val_dataset = CNN_dataset(dataframe = val_df,
                                    config = config,
                                    tokenizer = tokenizer)
    if config["task"] == "squadqa":
        train_dataset = SquadQADataset(dataframe = train_df,
                                      config = config,
                                      tokenizer = tokenizer)
        
        val_dataset = SquadQADataset(dataframe = val_df,
                                    config = config,
                                    tokenizer = tokenizer)
    if config["task"] == "europal":
        train_dataset = EuroPalDataset(dataframe = train_df,
                                      config = config,
                                      tokenizer = tokenizer)
        
        val_dataset = EuroPalDataset(dataframe = val_df,
                                    config = config,
                                    tokenizer = tokenizer)

    
    train_dl = DataLoader(train_dataset,
                            batch_size = config["training"]["batch_size"],
                            shuffle = True,
                            num_workers = config["num_workers"])
    
    val_dl = DataLoader(val_dataset,
                        batch_size = config["validation"]["batch_size"],
                        shuffle = True,
                        num_workers = config["num_workers"])
    
    
    #Freezing the model's parameters.

    for param in model.parameters():
        param.requires_grad = False

    soft_wte = softprompting(wte = model.get_input_embeddings(),
                             tokenizer = tokenizer,
                             prompt = config["prompt"])
    
    #Replacing the Model's parameters with the softembedding!
    model.set_input_embeddings(soft_wte)
    
    total_model_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Model Parameters: {total_model_params}")
    print(f"Total Trainable Parameters: {total_trainable_params}")

    optimizer = Adam(model.parameters(), lr = config["training"]["lr"])

    train(model = model,
          tokenizer = tokenizer,
          optimizer = optimizer,
          train_dataloader = train_dl,
          val_dataloader = val_dl,
          device = device,
          config = config,
          model_name = config["model_name"])

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hf_logging.set_verbosity_error()
    config = OmegaConf.load("config/config.yaml")
    overrides = OmegaConf.from_cli()
    config = OmegaConf.merge(config, overrides)
    OmegaConf.to_container (config)
    task = config["task"]
    config["wandb_run_name"] = config[task]["wandb_run_name"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config["wandb_logging"]:
        wandb.init(
            project = config["wandb_project"],
            name = config["wandb_run_name"],
            # config = config
        )
    trigger_training(config, "model_name", device)