import torch
from data.load_from_files import load_data
from data.preprocessing import preprocess_text, convert_to_idx
from transformer_model import transformer_model
from data.dataset import Dataset
from utils.utils import train, test, calculate_metrics, evaluate
from data.imports import *


def main(config):
    # load data
    train_data, dev_data, test_data = load_data()
    # preprocess data
    en_word_to_idx, en_idx_to_word, fr_word_to_idx, fr_idx_to_word, cleaned_train_data = preprocess_text(
        train_data, config)
    _, _, _, _, cleaned_dev_data = preprocess_text(dev_data, config)
    _, _, _, _, cleaned_test_data = preprocess_text(test_data, config)
    # add things to config file :-)
    config["n_src_vocab"] = len(en_word_to_idx)
    config["n_trg_vocab"] = len(fr_word_to_idx)
    config["en_word_to_idx"] = en_word_to_idx
    config["fr_word_to_idx"] = fr_word_to_idx
    config["fr_idx_to_word"] = fr_idx_to_word
    config["en_idx_to_word"] = en_idx_to_word
    config["pad_idx"] = en_word_to_idx["<pad>"]
    base_path = "/ssd_scratch/cvit/kolubex/"+wandb.run.name + "_"
    config["base_path"] = base_path
    # convert data to idx
    # print(f"Converting to indices")
    en_train_data, fr_train_data = convert_to_idx(
        cleaned_train_data, en_word_to_idx, fr_word_to_idx)
    en_dev_data, fr_dev_data = convert_to_idx(
        cleaned_dev_data, en_word_to_idx, fr_word_to_idx)
    en_test_data, fr_test_data = convert_to_idx(
        cleaned_test_data, en_word_to_idx, fr_word_to_idx)
    # print(f"Converted to indices")
    train_dataset = Dataset(en_train_data, fr_train_data)
    dev_dataset = Dataset(en_dev_data, fr_dev_data)
    test_dataset = Dataset(en_test_data, fr_test_data)

    # create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=False)
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=config["batch_size"], shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False)
    model = transformer_model(config).to(config["device"])
    # create optimizer with Adam with ROL pleateau scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.25, verbose=True, threshold=0.05)
    # create loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    # train model
    model = train(model, train_dataloader, dev_dataloader,
                  optimizer, scheduler, criterion, config)
    model = model.to(config["device"])
    # write things to files
    train_metrics = evaluate(model, train_dataloader, criterion, config,
                         write_to_file=True, file_name=base_path+"train.txt", type="train")
    dev_metrics = evaluate(model, dev_dataloader, criterion, config,
                          write_to_file=True, file_name=base_path+"dev.txt", type="dev")
    # save config as a .pt file
    pickle.dump(config, open(f"{base_path}config_{config['model_name']}.pt", "wb"))
    # test model
    metrics = evaluate(model, test_dataloader, criterion, config,
                   write_to_file=True, file_name=base_path + "test.txt", type="test")
    wandb.log(
        {
            "test_bleu_1": metrics[1]["test_bleu_1"],
            "test_bleu_2": metrics[1]["test_bleu_2"],
            "test_bleu_3": metrics[1]["test_bleu_3"],
            "test_loss": metrics[0],
        }
    )
    # wandb.log(metrics)
    model = model.to(config["device"])
    return model

def sweep_agent_manager():
    wandb.init()
    config = dict(wandb.config)
    print(config)
    config['model_name'] = f"Transformer_{config['d_model']}_{config['n_head']}_{config['batch_size']}_{config['lr']}"
    run_name = config['model_name']
    wandb.run.name = run_name
    main(config)

if __name__ == "__main__":
    config = {
        "seq_len": 20,
        "n_layers": 1,
        "n_head": 1,
        "d_k": 64,
        "d_v": 64,
        "d_model": 128,
        "d_inner": 256,
        "dropout": 0.1,
        "n_position": 200,
        "batch_size": 32,
        "lr": 0.001,
        "epochs": 10,
        "device": "cuda",
        "bleu_n": [1, 2, 3]
    }
    # initialize a wandb sweep
    sweep_config = {
        "method": "grid",
        "name": "transformer_sweep",
        "metric": {"goal": "maximize", "name": "val_bleu_1"},
        "parameters": {
            "d_model": {"values": [256, 512]},
            "n_head": {"values": [2, 4]},
            "seq_len": {"value": 20},
            "n_layers": {"value": 1},
            "d_k": {"value": 64},
            "d_v": {"value": 64},
            "d_inner": {"value": 128},
            "dropout": {"value": 0.1},
            "n_position": {"value": 200},
            "batch_size": {"value": 64},
            "lr": {"value": 0.001},
            "epochs": {"value": 30},
            "device": {"value": "cuda"},
            "bleu_n": {"value": [1, 2, 3]}
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="transformer", entity="lakshmipathi-balaji")
    # wandb.init(project="transformer", entity="lakshmipathi-balaji", name="transformer_128_1_32_0.001")
    # initialize a agent and get the config
    config = wandb.agent(sweep_id, function=sweep_agent_manager)
    # initalize a wandb run
    # config = dict(wandb.config)
    # print(config)
    # main(config)