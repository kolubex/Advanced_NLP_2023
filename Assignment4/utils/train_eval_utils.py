import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2LMHeadModel
import numpy as np
import wandb
import json
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from statistics import fmean
from  utils.utils import AverageMeter
from pathlib import Path
from torch.nn.utils import clip_grad_norm_
from .utils import EarlyStopping

def get_model(model_name="gpt2"):
        """
        Returns the model for the given model name.
        """

        if model_name == "gpt2":
            model = GPT2LMHeadModel.from_pretrained(model_name)
        return model

def get_tokenizer(model_name="gpt2"):
    """
    Returns the tokenizer for the given model name.
    """
    return AutoTokenizer.from_pretrained(model_name)

def calculate_bleu_scores(predicted_list, reference_list):
    bleu_scores = []

    for predicted, reference in zip(predicted_list, reference_list):
        predicted_tokens = predicted.split()
        reference_tokens = reference.split()
        score = sentence_bleu([reference_tokens], predicted_tokens)
        bleu_scores .append(score)

    return bleu_scores

@torch.no_grad()
def evaluate(model, val_dataloader, device, tokenizer):
    model.eval()
    loss_meter = AverageMeter("eval_loss", ":.5f")
    bleu_meter_val = AverageMeter("bleu_score_val", ":.5f")
    for data in tqdm(val_dataloader):
        article, label, label_text = data
        for key in article:
            article[key] = article[key].to(device)
        loss = model(**article, labels = label["input_ids"].to(device)).loss
        loss_meter.update(loss.item(), article["input_ids"].size(0))
        output = model.generate(input_ids = article["input_ids"].squeeze(1), attention_mask = article["attention_mask"].squeeze(1), eos_token_id = tokenizer.eos_token_id, repetition_penalty = 1.5, pad_token_id=tokenizer.eos_token_id)
        output = tokenizer.batch_decode(output, skip_special_tokens=True)
        
    
        val_bleu =  fmean(calculate_bleu_scores(output, list(label_text)))
        bleu_meter_val.update(val_bleu, article["input_ids"].size(0))

    return loss_meter.avg, bleu_meter_val.avg


def get_model_name(config):
    embed_size = config["embed_size"]
    num_layers = config["num_layers"]
    heads = config["heads"]
    dropout = config["dropout"]

    model_name = f"ES_{embed_size}_NL_{num_layers}_HD_{heads}_DP_{dropout}.pt"
    return model_name

def train(model, tokenizer, optimizer, train_dataloader, val_dataloader, device, config, model_name):
    model = model.to(device)
    epochs = config["epochs"]
    if config["early_stop"]["use"]:
        early_stopping = EarlyStopping(patience=config["early_stop"]["patience"],
                                       path=config["model_save_path"]+"early_ckp.pt",
                                       verbose=True)
        
    for epoch in tqdm(range(epochs), desc=f"Training Started"):
        model.train()
        loss = torch.tensor(0.0, device=device)
        loss_meter = AverageMeter("train_loss", ":.5f")
        bleu_meter_train = AverageMeter("bleu_score_train", ":.5f")
        if config["gradient_accumulate"]:
            accum_counter = 0
        for data in tqdm(train_dataloader):
            article, label, label_text = data
            for key in article:
                article[key] = article[key].to(device)
            loss += model(**article, labels = label["input_ids"].to(device)).loss
            output = model.generate(input_ids = article["input_ids"].squeeze(1), attention_mask = article["attention_mask"].squeeze(1), eos_token_id = tokenizer.eos_token_id, repetition_penalty = 1.5, pad_token_id=tokenizer.eos_token_id)
            output = tokenizer.batch_decode(output, skip_special_tokens=True)
            accum_counter += 1
            if config["gradient_accumulate"] and accum_counter % config["num_accumulate"] == 0:
                loss_meter.update(loss.item(), article["input_ids"].size(0))
                if config["clip_grads"]:
                    clip_grad_norm_(model.parameters(), max_norm = 1.0)
                loss.backward()
                optimizer.step()
                accum_counter = 0
                optimizer.zero_grad()
                loss = 0
            train_bleu =  fmean(calculate_bleu_scores(output, list(label_text)))
            bleu_meter_train.update(train_bleu, article["input_ids"].size(0))

        val_loss, val_bleu = evaluate(model, val_dataloader, device, tokenizer)
        #saving the model
        train_bleu_score = bleu_meter_train.avg
        train_loss = loss_meter.avg
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Train BLEU: {train_bleu_score}, Val Loss: {val_loss}, Val BLEU: {val_bleu}")
        data_to_log = {"epoch":  epoch, "train_loss": train_loss, "train_bleu": train_bleu_score, "val_loss": val_loss, "val_bleu": val_bleu}

        if config["wandb_logging"]:
            wandb.log(data_to_log)
        if epoch % config["model_save_freq"] == 0:
            path = Path(config["model_save_path"])
            if not path.exists():
                path.mkdir(parents=True)
            torch.save(model, config["model_save_path"]+"model.pt")
        
        if config["early_stop"]["use"]:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                #stop wandb
                if config["wandb_logging"]:
                    wandb.finish()
                print("Early stopping")
                break

