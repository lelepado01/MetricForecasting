
import os
import json
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

def get_metric(data, metric, time_incremental=False):
    try: 
        epochs = eval(data["entity"][metric]["prov-ml:metric_epoch_list"])
        values = eval(data["entity"][metric]["prov-ml:metric_value_list"])
        times = eval(data["entity"][metric]["prov-ml:metric_timestamp_list"])
    except: 
        return pd.DataFrame(columns=["epoch", "value", "time"])
    
    df = pd.DataFrame({"epoch": epochs, "value": values, "time": times}).drop_duplicates()

    if time_incremental: 
        df["time"] = df["time"].diff().fillna(0)

    df = df.sort_values(by="time")
    return df

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x), reverse=True)
    padded = pad_sequence(batch, batch_first=True)
    lengths = torch.tensor([len(seq) for seq in batch])
    return padded, lengths

class ProvJsonDataset(Dataset): 

    def __init__(self):
        super().__init__()

        self.path = "/Users/gabrielepadovani/Desktop/Università/Prov/yProv4MLProvenanceDataset/data/"
        self.files = [self.path + f for f in os.listdir(self.path)]
        fs = []
        for f in self.files: 
            data = json.load(open(f))
            if len(get_metric(data, "Loss_Context.TRAINING")["value"]) > 0: 
                fs.append(f)
        self.files = fs

    def __getitem__(self, index):
        data = json.load(open(self.files[index]))
        m = get_metric(data, "Loss_Context.TRAINING")["value"]
        # we want to predict the tendency, so the next steps knowing the starting point, 
        # essentially we train the model on the delta between the initial and the future points
        m = torch.tensor(m).float()
        # return (m - m.min()) / (m.max() - m.min())
        return m.diff()
    
    def __len__(self): 
        return len(self.files)