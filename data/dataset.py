import torch
from torch.utils.data import Dataset

class RoleDataset(Dataset):
    def __init__(self, dataframe, is_test = False):
        """
        Dataset class for role classification using pre-tokenized data.

        Args:
            dataframe (pd.DataFrame): DataFrame containing pre-tokenized inputs and labels.
        """
        self.dataframe = dataframe
        self.is_test = is_test

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieve an individual sample from the dataset.
        """
        row = self.dataframe.iloc[idx]

    
        item = {
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "File": row["File"],
            "Original Entity": row["Original Entity"],
            "Start": row["Start"],
            "End": row["End"]
        }
        if not self.is_test:
            item["labels"] = row["labels"]
            item["main_role_label"] = row["main_role_label"]
        return item