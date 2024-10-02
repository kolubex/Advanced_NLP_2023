## 1. File structure
The file structure is as follows:
```
.
├── config
│   └── config.yaml
├── dataloader
│   ├── cnn_dataset.py
│   ├── europal_dataset.py
│   ├── __pycache__
│   │   ├── cnn_dataset.cpython-310.pyc
│   │   ├── europal_dataset.cpython-310.pyc
│   │   └── squad_dataset.cpython-310.pyc
│   └── squad_dataset.py
├── README.md
├── Report.pdf
├── trainer.py
└── utils
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-310.pyc
    │   ├── softprompting.cpython-310.pyc
    │   ├── train_eval_utils.cpython-310.pyc
    │   └── utils.cpython-310.pyc
    ├── softprompting.py
    ├── train_eval_utils.py
    └── utils.py

5 directories, 18 files

```

## 2. How to run the code
* Run the code.
```bash
python trainer.py  # run with appropriate confg
```
* You can find the models and embeddings that are saved here: [Files_and_models](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/lakshmipathi_balaji_research_iiit_ac_in/EkXXvvvcZApFkWQRo0RCS8QBI_AJgbI1u3rXS3v1ww6GBQ?e=by4lRM)
* If you want to load them and run just use the following code.
* Link to WandB results: [WandB](https://wandb.ai/lakshmipathi-balaji/gptprompttune?workspace=user-lakshmipathi-balaji)