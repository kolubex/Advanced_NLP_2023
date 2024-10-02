from data.imports import *

# train function
def train(model, train_dataloader, dev_dataloader, optimizer, scheduler, criterion, config):
    outputs = []
    targets = []
    sources = []
    best_model = None
    best_bleu_1_gram = 0
    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0
        for src, trg in train_dataloader:
            src = src.to(config["device"])
            trg = trg.to(config["device"])
            optimizer.zero_grad()
            output = model(src, trg)
            # print all shapes
            # append to outputs and targets sample wise
            for i in range(output.shape[0]):
                sources.append(src[i,:].detach().cpu())
                # outputs.append(output[i,:-1,:].detach().cpu())
                # apply argmax over the last dimension and append as above
                outputs.append(torch.argmax(output[i,:-1,:], dim=-1).detach().cpu())
                targets.append(trg[i,1:].detach().cpu())
            output = output[:, :-1].reshape(-1, output.shape[-1])
            trg = trg[:,1:].reshape(-1)
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch: {epoch}, Train Loss: {epoch_loss/len(train_dataloader)}")
        scheduler.step(epoch_loss/len(train_dataloader))
        train_n_gram_bleu = calculate_metrics(sources, outputs, targets, config,type="train")
        # evaluate on dev set
        print(f"Train BLEU Scores: {train_n_gram_bleu}")
        dev_loss,val_n_gram_bleu  = evaluate(model, dev_dataloader, criterion, config, type="val")
        print(f"Epoch: {epoch}, Dev Loss: {dev_loss}")
        print(f"Dev BLEU Scores: {val_n_gram_bleu}")

        # log to wandb
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": epoch_loss/len(train_dataloader),
                "dev_loss": dev_loss,
                "train_bleu_1": train_n_gram_bleu["train_bleu_1"],
                "train_bleu_2": train_n_gram_bleu["train_bleu_2"],
                "train_bleu_3": train_n_gram_bleu["train_bleu_3"],
                "val_bleu_1": val_n_gram_bleu["val_bleu_1"],
                "val_bleu_2": val_n_gram_bleu["val_bleu_2"],
                "val_bleu_3": val_n_gram_bleu["val_bleu_3"],
                "lr": optimizer.param_groups[0]["lr"]
            }
        )
        if best_model is None or val_n_gram_bleu["val_bleu_1"] > best_bleu_1_gram:
            best_model = copy.deepcopy(model).cpu()
            best_bleu_1_gram = val_n_gram_bleu["val_bleu_1"]
            # save model
            print("Saving best model")
            torch.save(best_model, f"{config['base_path']}_best_model.pt")
    return best_model


# calculate metrics
def calculate_metrics(sources, outputs, targets, config, write_to_file=False,file_name="test.txt",type="test"):
    """ 
    Calculate BLEU Scores for n size given in list of config["bleu_n"]
    Args:
        outputs: list of tensors of shape (seq_len, batch_size, trg_vocab_size)
        targets: list of tensors of shape (seq_len, batch_size)
    Returns:
        None
    """
    # argmax over the last dimension
    # outputs = [torch.argmax(output, dim=-1) for output in outputs]
    written = False
    n_gram_bleu_scores = {}
    for n in config["bleu_n"]:
        for i in range(len(outputs)):
            sentence_bleu_scores = []
            hypothesis = outputs[i].tolist()
            # take the hypothesis until the first <eos> token
            if config["en_word_to_idx"]["<eos>"] in hypothesis:
                hypothesis = hypothesis[:hypothesis.index(config["en_word_to_idx"]["<eos>"])]
            # remove <pad> token, <sos> token, <eos> token => 0,1,2 from both hypothesis and targets
            hypothesis = [idx for idx in hypothesis if idx not in [0,1,2]]
            reference = targets[i].tolist()
            if config["fr_word_to_idx"]["<eos>"] in reference:
                reference = reference[:reference.index(config["fr_word_to_idx"]["<eos>"])]
            reference = [idx for idx in reference if idx not in [0,1,2]]
            source_sentence = sources[i].tolist()
            if config["en_word_to_idx"]["<eos>"] in source_sentence:
                source_sentence = source_sentence[:source_sentence.index(config["en_word_to_idx"]["<eos>"])]
            source_sentence = [idx for idx in source_sentence if idx not in [0,1,2]]
            bleu = sentence_bleu([reference], hypothesis, weights=(1/n,)*n)
            sentence_bleu_scores.append(bleu)
            if write_to_file and file_name and not written: 
                with open(file_name, "a") as f:
                    # print sources, predictions, bleu score
                    # line = source_sentence[i] + "\t" + hypothesis[i] + "\t" + str(bleu) + "\n"
                    # print after converting to words
                    source_sentence = [config["en_idx_to_word"][idx] for idx in source_sentence]
                    reference = [config["fr_idx_to_word"][idx] for idx in reference]
                    hypothesis = [config["fr_idx_to_word"][idx] for idx in hypothesis]
                    line = " ".join(source_sentence) + "\t" + " ".join(hypothesis) +"\t"+ " ".join(reference)+ "\t" + str(bleu) + "\n"
                    f.write(line)
        written = True
        n_gram_bleu_scores[type + '_bleu_' + str(n)] = np.mean(sentence_bleu_scores)
    return n_gram_bleu_scores



# test function
def evaluate(model, test_dataloader, criterion, config, write_to_file=False, file_name="test.txt",type="test"):
    """ 
    It doesn't work as it is knowing the real word right, so check evaluate function. 
    """
    outputs = []
    targets = []
    sources = []
    model.eval()
    val_loss = 0
    for src, trg in test_dataloader:
        src = src.to(config["device"])
        trg = trg.to(config["device"])
        output = model(src, trg)
        # append to outputs and targets sample wise
        for i in range(output.shape[0]):
            sources.append(src[i,:].detach().cpu())
            # outputs.append(output[i,:-1,:].detach().cpu())
            # apply argmax over the last dimension and append as above
            outputs.append(torch.argmax(output[i,:-1,:], dim=-1).detach().cpu())
            targets.append(trg[i,1:].detach().cpu())
        output = output[:, :-1].reshape(-1, output.shape[-1])
        trg = trg[:,1:].reshape(-1)
        loss = criterion(output, trg)
        val_loss += loss.item()
    test_n_gram_bleu = calculate_metrics(sources, outputs, targets, config, write_to_file, file_name, type=type)
    return val_loss/len(test_dataloader), test_n_gram_bleu

def evaluate(model, test_dataloader, criterion, config, write_to_file=False, file_name="test.txt", type="test"):
    outputs = []
    targets = []
    sources = []
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src, trg in test_dataloader:
            src = src.to(config["device"])
            trg = trg.to(config["device"])
            output = model.generate(src, config, max_len= config["seq_len"])  # Use a generation method in your model
            # Append to outputs and targets sample-wise
            for i in range(output.shape[0]):
                sources.append(src[i, :].detach().cpu())
                outputs.append(output[i, :].detach().cpu())
                targets.append(trg[i, 1:].detach().cpu())
            output = output[:, 1:].reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            val_loss += loss.item()
    test_n_gram_bleu = calculate_metrics(sources, outputs, targets, config, write_to_file, file_name, type=type)
    return val_loss / len(test_dataloader), test_n_gram_bleu