## 1. File structure
The file structure is as follows:
```
.
├── README.md
├── Report.pdf
└── src
    ├── data
    │   ├── dataset.py
    │   ├── imports.py
    │   ├── load_from_files.py
    │   └── preprocessing.py
    ├── main.py
    ├── models
    │   ├── layers.py
    │   ├── models.py
    │   ├── modules.py
    │   └── sublayers.py
    ├── transformer_model.py
    └── utils
        └── utils.py

4 directories, 13 files
```

## 2. How to run the code
* Change directoy to src directory.
```
cd src
```
* Run the code.
```bash
python main.py  # Initialises a sweep so keep your entity name.
```
* You can find the models and embeddings that are saved here: [Files_and_models](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/lakshmipathi_balaji_research_iiit_ac_in/EmDl-MM1i1NBkdLTt86xJWcBVjLvcxwBMmrwKnji8gyYrg?e=Mw6IBm)
* If you want to load them and run just use the following code.
* Link to WandB results: [WandB](https://wandb.ai/lakshmipathi-balaji/transformer/sweeps/sx5qymdj?workspace=user-lakshmipathi-balaji)