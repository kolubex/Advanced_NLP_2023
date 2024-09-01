
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

def get_embeddings(config,vocab):
    """
    Args: 
        config: configuration of the model
    Returns:
        embedding_dict: dictionary with word as key and embedding as value
    """
    print("Loading embeddings...")

    embedding_size = config['embedding_dim']
    embeddings = {}
    if (embedding_size == 300):
        model = "fasttext-wiki-news-subwords-300"
    elif (embedding_size == 200):
        model = "glove-wiki-gigaword-200"
    model = api.load(model)
    print(f"Model loaded: {model}")
    words = list(model.index_to_key)
    for word in words:
        if word in vocab:
            embeddings[word] = torch.from_numpy(model[word]).to(config['device'])
    print(f"Number of words in the embeddings: {len(embeddings.keys())}")
    return embeddings

# config = {
#     "epochs": 10,
#     "lr": 0.001,
#     "batch_size": 32,
#     "optimizer": "adam",
#     "context_size": 5,
#     "embedding_dim": 300,
#     "hidden_dim": 300,
#     "dropout": 0.2,
#     "device": "cuda"
# }
class NNLM(nn.Module):
    """
    NNLM model
    Returns:
        output: vector with probabilities for each word in the vocabulary
    """

    def __init__(self, config):
        """
        Args:
            vocab_size: size of the vocabulary
            embedding_dim: dimension of the embedding
            hidden_dim: dimension of the hidden layer
            context_size: size of the context
            sentence_length: length of the sentence
        """
        super(NNLM, self).__init__()
        self.vocab_size = config["vocab_size"]
        self.embedding_dim = config["embedding_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.context_size = config["context_size"]
        self.device = config["device"]
        self.dropout = nn.Dropout(config["dropout"])
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.hidden_layer = nn.Linear(self.context_size * self.embedding_dim, self.hidden_dim)
        # relu
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(self.hidden_dim, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        """
        Args:
            list sentences (batch_size, one_hot_vectors_of_words)
        Returns:
            output: vector with probabilities for each word in the vocabulary
        """
        # start from the first 4 words predict the 5th word, by passing the embeddings through the hidden layer
        # these embeddings are obtained from the embedding layer
        # the output of the hidden layer is passed to a layer with prev_hidden_dim*vocab_size neurons
        # the output of this layer is passed to a softmax layer with vocab_size neurons
        # the output of the softmax layer is the probability of each word in the vocabulary
        outputs = []
        for i in range(len(x)):
            for j in range(self.context_size):
                embeds = x[i-j].to(self.device)
                embeds = embeds.view(1, -1).to(self.device)
                if j == 0:
                    embeds_concat = embeds.to(self.device)
                else:
                    embeds_concat = torch.cat((embeds_concat, embeds), 1).to(self.device)
            # pass the embeddings to the hidden layer
            hidden = self.hidden_layer(embeds_concat)
            hidden = self.dropout(hidden)
            hidden = self.relu(hidden)
            # pass the output of the hidden layer to the output layer
            output = self.output_layer(hidden)
            output = self.dropout(output)
            # pass the output of the output layer to the softmax layer
            output = self.softmax(output)
            outputs.append(output)
        # convert the list to a tensor and squeeze the second dimension
        outputs = torch.stack(outputs).squeeze(1)
        return outputs
    

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
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0
    # get batches from the dataloader
    perplexity = 0
    batch_num = 0
    # with model in evaluation mode
    model.eval()
    for batch in dataloader:
        # get the input and target batch
        input_batch = batch[0]
        target_batch = batch[1]
        # get the output of the model
        output = model(input_batch)
        # calculate the loss
        loss = loss_function(output, target_batch)
        total_loss += loss.item()
        batch_num += 1
    perplexity = np.exp(total_loss/batch_num)
    return perplexity, total_loss/batch_num

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
            # print(f"Batch number: {batch_num}")
            # get the input and target batch
            input_batch = batch[0]
            target_batch = batch[1]
            # zero the gradients
            model.zero_grad()
            # get the output of the model
            output = model(input_batch)
            # calculate the loss
            # print(f"Output shape: {output.shape}")
            # print(f"Target batch shape: {target_batch.shape}")
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
        # mkdir if not exists /ssd_scratch/cvit/kolubex_anlp/
        if not os.path.exists("/ssd_scratch/cvit/kolubex_anlp/"):
            os.makedirs("/ssd_scratch/cvit/kolubex_anlp/")
        # save the model
        torch.save(model.state_dict(), f"/ssd_scratch/cvit/kolubex_anlp/{config['model_name']}.pth")
        # wandb.log(data_to_log)
    return model

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
        self.context_size = config['context_size']
        self.device = config['device']
        self.embeddings = embeddings

    def __getitem__(self, index):
        batch = self.data[index]
        unk_tensor = self.embeddings["unk"].to(self.device)
        input_batch = [unk_tensor*(self.context_size-1)]
        target_batch = []
        for sentence in batch:
            words = sentence.split()
            for word in words:
                if(word not in self.word_to_ix.keys()):
                    input_batch.append(unk_tensor.to(self.device))
                    target_batch.append(torch.tensor(self.word_to_ix["unk"]).to(self.device))
                elif(word in self.embeddings.keys()):
                    input_batch.append((self.embeddings[word]).to(self.device))
                    target_batch.append(torch.tensor(self.word_to_ix[word]).to(self.device))
                else:
                    input_batch.append(unk_tensor.to(self.device))
                    target_batch.append(torch.tensor(self.word_to_ix["unk"]).to(self.device))
        # remove the last element from the input batch tensor
        input_batch = input_batch[:-1]
        input_batch = torch.stack(input_batch).to(self.device)
        target_batch = torch.stack(target_batch).to(self.device)
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
    test_data = np.random.choice(sentences, num_test, replace=False)
    num_val = 10000
    val_data = np.random.choice(sentences, num_val, replace=False)
    # the rest of the sentences are for training by removing the test and validation sentences
    train_data = list(set(sentences) - set(test_data) - set(val_data))
    # create the vocabulary
    vocab = create_vocab(train_data)
    # create the word_to_ix dictionary
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    # create the ix_to_word dictionary
    ix_to_word = {i: word for i, word in enumerate(vocab)}
    # get embeddings
    embeddings = get_embeddings(config, vocab)
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
    # create the model
    # add the vocab_size to the config
    config['vocab_size'] = vocab_size
    model = NNLM(config)
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
    config['model_name'] = f"NNLM_{config['optimizer']}_lr_{config['lr']}_batch_size_{config['batch_size']}_epochs_{config['epochs']}"
    run_name = config['model_name']
    wandb.run.name = run_name
    main(config)


config = {
    "epochs": 10,
    "lr": 0.001,
    "batch_size": 512,
    "optimizer": "adam",
    "context_size": 5,
    "embedding_dim": 200,
    "hidden_dim": 300,
    "dropout": 0.2,
    "device": "cuda"
}
print(config["dropout"])
main(config)

# if __name__ == "__main__":
#     wandb.login()
#     wandb.agent(sweep_id="lakshmipathi-balaji/anlp/3y0uss52", function=sweep_agent_manager, count=100)