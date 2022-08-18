from torch.utils.data import Dataset
import pandas as pd


class SmilesDataset(Dataset):
    def __init__(self, filename):
        self.data = pd.read_csv(filename)["Smiles"]

    def __len__(self):
        return self.data.count()

    def __getitem__(self, idx):
        return self.data[idx]
