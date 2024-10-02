
# make a function that has the data frame of form:
# 	french	english
# 0	Quand j'avais la vingtaine, j'ai vu mes tout p...	When I was in my 20s, I saw my very first psyc...
# 1	J'étais étudiante en thèse en psychologie clin...	I was a Ph.D. student in clinical psychology a...

# it reuturn en_word_to_idx, en_idx_to_word, fr_word_to_idx, fr_idx_to_word, cleaned_data in every row with seq length 20.
# add <sos> and <eos> to the beginning and end of each sentence.
# if length is more than 20 then crop it to 20, if less than 20 then pad it with 0.
# the word_to_idx should have <pad> as 0, <sos> as 1, <eos> as 2, <unk> as 3.

import torch
from .imports import *

def preprocess_text(data, config):

    seq_len = config["seq_len"]
    # Initialize special tokens
    special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
    
    # Create initial dictionaries with special tokens
    en_word_to_idx = {word: idx for idx, word in enumerate(special_tokens)}
    en_idx_to_word = {idx: word for idx, word in enumerate(special_tokens)}
    fr_word_to_idx = {word: idx for idx, word in enumerate(special_tokens)}
    fr_idx_to_word = {idx: word for idx, word in enumerate(special_tokens)}
    
    # Initialize cleaned_data
    cleaned_data = []
    for idx, row in data.iterrows():
        en_sentence = row['english'].lower()
        fr_sentence = row['french'].lower()
        en_sentence = re.sub(r'[^\w\s<>]', r' ', en_sentence)
        fr_sentence = re.sub(r'[^\w\s<>]', r' ', fr_sentence)
        en_tokens = nltk.word_tokenize(en_sentence, language='english')
        fr_tokens = nltk.word_tokenize(fr_sentence, language='french')
        if len(en_tokens):
            en_sentence= " ".join(en_tokens)
        else:
            en_sentence= ""
        if len(fr_tokens):
            fr_sentence= " ".join(fr_tokens)
        else:
            fr_sentence= ""
        # Tokenize the English and French sentences
        en_tokens = ["<sos>"] + en_sentence.split()[:(seq_len-2)] + ["<eos>"]
        fr_tokens = ["<sos>"] + fr_sentence.split()[:(seq_len-2)] + ["<eos>"]

        # Pad with <pad> if the sequence is shorter than seq_len
        en_tokens += ["<pad>"] * (seq_len - len(en_tokens))
        fr_tokens += ["<pad>"] * (seq_len - len(fr_tokens))

        # Update word-to-index and index-to-word dictionaries
        for token in en_tokens:
            if token not in en_word_to_idx:
                en_word_to_idx[token] = len(en_word_to_idx)
                en_idx_to_word[len(en_word_to_idx) - 1] = token

        for token in fr_tokens:
            if token not in fr_word_to_idx:
                fr_word_to_idx[token] = len(fr_word_to_idx)
                fr_idx_to_word[len(fr_word_to_idx) - 1] = token

        cleaned_data.append([en_tokens, fr_tokens])

    return en_word_to_idx, en_idx_to_word, fr_word_to_idx, fr_idx_to_word, cleaned_data

def convert_to_idx(cleaned_data, en_word_to_idx, fr_word_to_idx):
    en_data = []
    fr_data = []
    for pair in cleaned_data:
        en_sentence = []
        fr_sentence = []
        for word in pair[0]:
            en_sentence.append(en_word_to_idx[word] if word in en_word_to_idx else en_word_to_idx["<unk>"])
        for word in pair[1]:
            fr_sentence.append(fr_word_to_idx[word] if word in fr_word_to_idx else fr_word_to_idx["<unk>"])
        # make them into torch tensors
        en_data.append(torch.LongTensor(en_sentence))
        fr_data.append(torch.LongTensor(fr_sentence))
    return en_data, fr_data