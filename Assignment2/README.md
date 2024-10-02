The file structure is as follows:
```
.
├── README.md
├── Report.pdf
└── src
    ├── classifier_nnparam.py
    ├── classifier.py
    ├── elmo.py
    └── utils
        ├── classifierr_model.py
        ├── data_classifier.py
        ├── data_elmo.py
        ├── imports.py
        └── preprocessing.py

2 directories, 10 files
```

## 2. How to run the code
* Install the requirements in a new environment.
```
pip install -r requirements.txt
```
* Change directoy to src directory.
```
cd src
```
* Run the code.
```bash
python elmo.py  # For ELMO pretraining
python classifier.py  # For downstream task
# Note that we model has to be downloaded and stored in the correct folder to run the code, and data has to be respective folder (data) in the same directory as the code.
# And please initalise with a sweep in wandb if you want to run the code without wandb then run only main function in the code.

```
* The models need for the embeddings will be autodownloaded when you run the code.
* Please mention the path that you want the model to be save in the code.
* You can find the models and embeddings that are saved here: [Models](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/lakshmipathi_balaji_research_iiit_ac_in/EvtTVQTb2DVIsoGL6_g9yroB4oaKe6OIiVe6VVP8AkiUkw?e=TiSwZZ)
* If you want to load them and run just use the following code.
```py
import torch
model = torch.load('path_to_model')
# given it is inside the script given in the src folder as it needs the class definition.
```
* Link to WandB results: [WandB](https://wandb.ai/lakshmipathi-balaji/anlp_a2?workspace=user-lakshmipathi-balaji)