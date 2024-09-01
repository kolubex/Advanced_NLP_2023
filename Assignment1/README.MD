The file structure is as follows:
```
.
├── data
│   ├── Adv_NLP_Assignment_1.pdf
│   └── Auguste_Maquet.txt
├── perplexity_files
│   ├── 2021114007-LM1-test.txt
│   ├── 2021114007-LM1-train.txt
│   ├── 2021114007-LM1-val.txt
│   ├── 2021114007-LM2-test.txt
│   ├── 2021114007-LM2-train.txt
│   ├── 2021114007-LM2-val.txt
│   ├── 2021114007-LM3-test.txt
│   ├── 2021114007-LM3-train.txt
│   └── 2021114007-LM3-val.txt
├── README.MD
├── Report.pdf
├── requirements.txt
└── src
    ├── lstm.py
    ├── nnlm.py
    └── transformer.py

3 directories, 17 files
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
```
python lstm.py  # For LSTM
python nnlm.py  # For NNLM
python transformer.py  # For Transformer
```
* The models need for the embeddings will be autodownloaded when you run the code.
* Please mention the path that you want the model to be save in the code.
* You can find the models that are saved here: [Models](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/lakshmipathi_balaji_research_iiit_ac_in/Eh5KGR7Vd-JCqezQD4u1nIQBuNxOqqmrFrZwjwaBCEAgEw?e=eFyIp6)
* If you want to load them and run just use the following code.
```py
import torch
model = torch.load('path_to_model')
```
* Link to WandB results: [WandB](https://wandb.ai/lakshmipathi-balaji/anlp?workspace=user-lakshmipathi-balaji)