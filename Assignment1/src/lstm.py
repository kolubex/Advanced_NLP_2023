import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import regex as re
import nltk
nltk.download('punkt')
import wandb
import gensim.downloader as api

def get_embeddings(config):
    """
    Args: 
        config: configuration of the model
    Returns:
        embedding_dict: dictionary with word as key and embedding as value
    """

    embedding_size = config['embedding_dim']
    embeddings = {}
    print("Loading embeddings...")
    if (embedding_size == 300):
        model = "fasttext-wiki-news-subwords-300"
    elif (embedding_size == 200):
        model = "glove-wiki-gigaword-200"
    model = api.load(model)
    words = list(model.index_to_key)
    for word in words:
        embeddings[word] = torch.from_numpy(model[word]).to(config['device'])
    print(f"Number of words in the embeddings: {len(embeddings.keys())}")
    return embeddings

def create_vocab(data):
    """
    Args:
        data: the data to create the vocabulary
    Returns:
        vocab: the vocabulary
    """
    vocab = ["unk", "sos", "eos"]
    for sentence in data:
        for word in sentence.split():
            if word not in vocab:
                vocab.append(word)
    
    # return the unique words in the vocabulary
    return list(set(vocab))



def preprocess(corpus):
    """
    Args:
        corpus: the corpus to preprocess (str)
    Returns:
        sentences: list of sentences
    """
    cleaned_sentences = []
    sentence_pattern = r'[.?!;]'
    # Use the re.split() function to split the text into sentences
    sentences = re.split(sentence_pattern, corpus)

    # Remove empty strings and leading/trailing whitespace from the sentences
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    for sentence in sentences:
        # make lowercase
        sentence = sentence.lower()
        text = re.sub(r'(!|"|\#|\$|%|”|“|&|\'|\(|\)|\*|\+|,|-|—|’|\.|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|‘|\{|\||\}|~{1,})', r' ', sentence)
        # tokenize the text
        tokens = nltk.word_tokenize(text)
        if(len(tokens)):
            # create a sentence with space separated tokens
            sentence = ' '.join(tokens)
            # add <s> and </s> to the sentence
            sentence = ' sos ' + sentence + ' eos '
            # append the sentence to the sentences list
            cleaned_sentences.append(sentence)
    return cleaned_sentences

def get_perplexity(model,config, dataloader):
    """
    Args:
        model: the model to train
        train_data: the training data
        epochs: number of epochs
        lr: learning rate
        batch_size: size of the batch
        config: configuration of the model
    Returns:
        perplexity: the perplexity of the model on the test data
    """
    batch_size = config['batch_size']
    epochs = config['epochs']
    lr = config['lr']
    seq_len = config['seq_len']
    loss_function = F.cross_entropy
    total_loss = 0
    # get batches from the dataloader
    perplexity = 0
    # with model in evaluation mode
    model.eval()
    batch_num = 0
    for batch in dataloader:
        # get the input and target batch
        input_batch = batch[0]
        target_batch = batch[1]
        # zero the gradients
        model.zero_grad()
        # get the output of the model
        output = model(input_batch)
        # calculate the loss
        output = output.view(-1, output.shape[2])
        target_batch = target_batch.view(-1)
        loss = loss_function(output, target_batch, ignore_index=config['word_to_ix']['eos'])
        total_loss += loss.item()
        # print(f"Loss: {loss.item()}")
        batch_num += 1
    perplexity = np.exp(total_loss/batch_num)
    print(f"Loss: {total_loss/batch_num}")
    return perplexity, total_loss/batch_num

# define dataset class
class Dataset(torch.utils.data.Dataset):
    """
    Dataset class
    TODO:
        * Rather than just passing the index of the word, pass the embedding 
        of the word from the ./embeddings.txt file.
    """
    def __init__(self, data, word_to_ix,config,embeddings):
        self.data = data
        self.word_to_ix = word_to_ix
        self.unk_index = word_to_ix['unk']
        self.eos_index = word_to_ix['eos']
        # self.context_size = config['context_size']
        self.embedding_dim = config['embedding_dim']
        self.seq_len = config['seq_len']
        self.device = config['device']
        self.embeddings = embeddings

    def __getitem__(self, index):
        batch = self.data[index]
        unk_tensor = self.embeddings["unk"].to(self.device)
        sos_tensor = self.embeddings["sos"].to(self.device)
        pad_tensor = torch.zeros(self.embedding_dim).to(self.device)
        sentence_input_batch = []
        sentence_target_batch = []
        for sentence in batch:
            input_batch = []
            target_batch = []
            words = sentence.split()
            i = 0
            for word in words:
                if(i<self.seq_len):
                    if(word not in self.word_to_ix.keys()):
                        input_batch.append(unk_tensor.to(self.device))
                        target_batch.append(torch.tensor(self.word_to_ix["unk"]).to(self.device))
                    elif(word in self.embeddings.keys()):
                        input_batch.append((self.embeddings[word]).to(self.device))
                        target_batch.append(torch.tensor(self.word_to_ix[word]).to(self.device))
                    else:
                        input_batch.append(unk_tensor.to(self.device))
                        target_batch.append(torch.tensor(self.word_to_ix["unk"]).to(self.device))
                i += 1
            input_batch = input_batch[:-1]
            target_batch = target_batch[1:]
            # if the sentence is less than the sequence length, pad it
            if(len(input_batch) < self.seq_len):
                input_batch = input_batch +  [pad_tensor for i in range(self.seq_len - len(input_batch))] 
                target_batch = target_batch + [torch.tensor(self.eos_index).to(self.device) for i in range(self.seq_len - len(target_batch))] 
            # stack the input and target batch
            input_batch = torch.stack(input_batch)
            target_batch = torch.stack(target_batch)
            sentence_input_batch.append(input_batch)
            sentence_target_batch.append(target_batch)
        # remove the last element from the input batch tensor
        input_batch = torch.stack(sentence_input_batch).to(self.device)
        target_batch = torch.stack(sentence_target_batch).to(self.device)
        return input_batch, target_batch

    def __len__(self):
        return len(self.data)

# define DataLoader
class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, num_workers=8):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.dataset.data)
        self.batch_num = 0
        return self

    def __next__(self):
        if self.batch_num * self.batch_size >= len(self.dataset):
            raise StopIteration
        batch = self.dataset[self.batch_num * self.batch_size: (self.batch_num + 1) * self.batch_size]
        self.batch_num += 1
        return batch

class LSTM(nn.Module):
     # give me a lstm model that is used for classification of a sentence
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.n_layers = config['num_layers']
        
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.n_layers)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.normalize = nn.Softmax(dim = 2)

    def forward(self, x):
        # x is a batch of input tensors
        # x.shape = (batch_size, sequence_length, embedding_dim)
        # output.shape = (batch_size, sequence_length, hidden_dim)
        output, (hidden, cell) = self.lstm(x)
        # print("Output shape: ", output.shape)
        # output.shape = (batch_size, sequence_length, output_dim)
        output = self.fc(output)
        # print("Output shape: ", output.shape)
        # output.shape = (batch_size, sequence_length, output_dim)
        output = self.normalize(output)
        return output


def train_nn(model, config, train_dataloader,val_dataloader,test_dataloader):
    """
    Args:
        model: the model to train
        train_data: the training data
        epochs: number of epochs
        lr: learning rate
        batch_size: size of the batch
        config: configuration of the model
    Returns:
        model: the trained model
    """
    batch_size = config['batch_size']
    epochs = config['epochs']
    lr = config['lr']
    loss_function = nn.CrossEntropyLoss()
    if config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    device = config['device']
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        batch_num = 0
        # get batches from the dataloader
        for batch in train_dataloader:
            # get the input and target batch
            input_batch = batch[0]
            target_batch = batch[1]
            # zero the gradients
            model.zero_grad()
            # get the output of the model
            output = model(input_batch)
            output = output.view(-1, output.shape[2])
            target_batch = target_batch.view(-1)
            loss = loss_function(output, target_batch)
            # backpropagate the loss
            loss.backward()
            # update the parameters
            optimizer.step()
            # add the loss to the total loss
            total_loss += loss.item()
            batch_num += 1
        print("Epoch: {}, Loss: {}".format(epoch, total_loss))
        # print perplexity on the validation set
        perplexity, loss = get_perplexity(model, config, val_dataloader)
        print(f"Perplexity on the validation set: {perplexity}")
        print(f"Loss on the validation set: {loss}")
        # train_perplexity, train_loss = get_perplexity(model, config, train_dataloader)
        # print(f"Perplexity on the training set: {train_perplexity}")
        # print(f"Loss on the training set: {train_loss}")
        test_preplexity, test_loss = get_perplexity(model, config, test_dataloader)
        print(f"Perplexity on the test set: {test_preplexity}")
        print(f"Loss on the test set: {test_loss}")
        data_to_log = {
            "epoch": epoch,
            "train_loss": total_loss,
            "val_loss": loss,
            "test_loss": test_loss,
            # "train_perplexity": train_perplexity,
            "val_perplexity": perplexity,
            "test_perplexity": test_preplexity            
        }
        print(data_to_log)
        # save the model
        # mkdir if not exists /ssd_scratch/cvit/kolubex_anlp_lstm/
        if not os.path.exists("/ssd_scratch/cvit/kolubex_anlp_lstm/"):
            os.makedirs("/ssd_scratch/cvit/kolubex_anlp_lstm/")
        # save the model
        torch.save(model.state_dict(), f"/ssd_scratch/cvit/kolubex_anlp_lstm/{config['model_name']}.pth")
        # wandb.log(data_to_log)
        print("Model saved!")
        print(data_to_log)
    return model

def main(config):
    # read the corpus
    with open('../data/Auguste_Maquet.txt', 'r') as f:
        corpus = f.read()
    # preprocess the data and get sentences back from preprocess()
    sentences = preprocess(corpus)
    print(f"Number of sentences: {len(sentences)}")
    # divide these sentences into train and test data with randomly 10000 
    # sentences for test and the rest for training
    num_test = 20000
    # randomly chose 10000 sentences for test
    test_data = np.random.choice(sentences, num_test, replace=False)
    num_val = 10000
    # randomly chose 10000 sentences for validation
    val_data = np.random.choice(sentences, num_val, replace=False)
    # the rest of the sentences are for training by removing the test and validation sentences
    train_data = list(set(sentences) - set(test_data) - set(val_data))
    # create the vocabulary
    vocab = create_vocab(train_data)
    config["output_dim"] = len(vocab)
    # create the word_to_ix dictionary
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    # create the ix_to_word dictionary
    ix_to_word = {i: word for i, word in enumerate(vocab)}
    # get embeddings
    embeddings = get_embeddings(config)
    print(f"Number of words in the vocabulary: {len(vocab)}")
    # create the dataset
    train_dataset = Dataset(train_data, word_to_ix,config,embeddings)
    val_dataset = Dataset(val_data, word_to_ix,config,embeddings)
    test_dataset = Dataset(test_data, word_to_ix,config,embeddings)
    # create the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)
    # get the vocab size
    vocab_size = len(vocab)
    config["word_to_ix"] = word_to_ix
    # create the model
    # add the vocab_size to the config
    config['vocab_size'] = vocab_size
    model = LSTM(config)
    # set the device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # config['device'] = device
    model.to(config['device'])
    # train the model
    model = train_nn(model, config, train_dataloader,val_dataloader,test_dataloader)
    # get the perplexity of the model on the test data
    perplexity,loss = get_perplexity(model, config, test_dataloader)
    print(f"Perplexity on the test set: {perplexity}")
    print(f"Loss on the test set: {loss}")

def sweep_agent_manager():
    wandb.init()
    config = dict(wandb.config)
    print(config)
    config['model_name'] = f"LSTM_{config['optimizer']}_lr_{config['lr']}_batch_size_{config['batch_size']}_epochs_{config['epochs']}_nl_{config['num_layers']}_hd_{config['hidden_dim']}_ed_{config['embedding_dim']}_sl_{config['seq_len']}"
    run_name = config['model_name']
    wandb.run.name = run_name
    main(config)

config = {
    'embedding_dim': 300,
    'hidden_dim': 128,
    'num_layers': 2,
    'optimizer': 'adam',
    'lr': 0.001,
    'batch_size': 64,
    'epochs': 25,
    'device': 'cuda',
    'seq_len': 20,
    'model_name': 'lstm_aditya'
}
main(config)

# if __name__ == "__main__":
#     wandb.login()
#     wandb.agent(sweep_id="lakshmipathi-balaji/anlp/30i3r7av", function=sweep_agent_manager, count=100)