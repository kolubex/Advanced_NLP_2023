
# load data from files

import os
import pandas as pd

def load_data():
    files = ["ted-talks-corpus/train.en", "ted-talks-corpus/train.fr","ted-talks-corpus/dev.en", "ted-talks-corpus/dev.fr", "ted-talks-corpus/test.en", "ted-talks-corpus/test.fr"]
    test_data = pd.DataFrame()
    train_data = pd.DataFrame()
    dev_data = pd.DataFrame()
    for file in files:
        if "train" in file:
            french_file = file[:-2] + "fr"
            english_file = file[:-2] + "en"
            train_data =  pd.DataFrame({"french": open(french_file, "r").readlines(), "english": open(english_file, "r").readlines()})
        elif "dev" in file:
            french_file = file[:-2] + "fr"
            english_file = file[:-2] + "en"
            dev_data =  pd.DataFrame({"french": open(french_file, "r").readlines(), "english": open(english_file, "r").readlines()})
        elif "test" in file:
            french_file = file[:-2] + "fr"
            english_file = file[:-2] + "en"
            test_data = pd.DataFrame({"french": open(french_file, "r").readlines(), "english": open(english_file, "r").readlines()})
    return train_data, dev_data, test_data
