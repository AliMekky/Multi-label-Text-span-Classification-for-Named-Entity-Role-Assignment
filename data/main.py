from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from data.preprocessing import load_and_tokenize_data
from data.dataset import RoleDataset

annotation_file = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/EN/subtask-1-annotations.txt"
documents_dir = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/EN/raw-documents"
save_path = "preprocessed_data.pt"


tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")


# preprocessed_data = torch.load(save_path) if os.path.exists(save_path) else load_and_tokenize_data(
#     annotation_file, documents_dir, tokenizer, save_path=save_path
# )
preprocessed_data = load_and_tokenize_data(annotation_file, documents_dir, tokenizer)
df, all_roles = preprocessed_data


dataset = RoleDataset(df)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print(len(dataset))
for batch in dataloader:
    print("Input IDs shape:", batch["input_ids"].shape)
    print("Attention Mask shape:", batch["attention_mask"].shape)
    print("Labels shape:", batch["labels"].shape)
    break