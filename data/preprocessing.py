import os
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from configs.config import LABELS, MAIN_ROLES


def load_and_tokenize_data(annotation_file, documents_dir, tokenizer, max_length=512, save_path=None, is_test=False):
    """
    Load annotations, preprocess the text, tokenize, and save the tokenized dataset.

    Args:
        annotation_file (str): Path to the annotation file.
        documents_dir (str): Directory containing the raw documents.
        tokenizer: Tokenizer instance (e.g., from Hugging Face).
        max_length (int): Maximum sequence length for tokenization.
        save_path (str, optional): Path to save the tokenized dataset. If None, the dataset is not saved.
        is_test (bool): Whether the dataset is a test set (does not include labels).

    Returns:
        pd.DataFrame: A DataFrame containing tokenized inputs and optionally labels for training data.
    """


    with open(annotation_file, "r", encoding="utf-8") as file:
        annotation_lines = file.readlines()

    structured_data = []

    for line in annotation_lines:
        parts = line.strip().split("\t")
        file_id = parts[0]
        entity = parts[1]
        start = int(parts[2])
        end = int(parts[3])
        roles = parts[4:] if not is_test else []

        document_path = os.path.join(documents_dir, file_id)
        if os.path.exists(document_path):
            with open(document_path, "r", encoding="utf-8") as doc_file:
                document = doc_file.read()
        else:
            print(f"Warning: File {file_id} not found.")
            document = ""

        paragraph, paragraph_start = extract_paragraph_containing_entity_with_offset(document, start, end)

        relative_start = start - paragraph_start
        relative_end = end - paragraph_start

        paragraph_with_entity = replace_entity_with_token(paragraph, relative_start, relative_end + 1, "<ENTITY>")

        tokenized_inputs = tokenizer(
            paragraph_with_entity,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        data_entry = {
            "input_ids": tokenized_inputs["input_ids"].squeeze(0),
            "attention_mask": tokenized_inputs["attention_mask"].squeeze(0),
            "Processed Paragraph": paragraph_with_entity,
            "File": file_id,
            "Original Entity": entity,
            "Start": start,
            "End": end
        }

        if not is_test:
            label_vector = torch.zeros(len(LABELS), dtype=torch.float)
            for role in roles:
                if role in LABELS:
                    role_index = LABELS.index(role)
                    label_vector[role_index] = 1
            data_entry["labels"] = label_vector
            data_entry["main_role_label"] = MAIN_ROLES.index(parts[4])
            

        structured_data.append(data_entry)

    df = pd.DataFrame(structured_data)

    if save_path:
        torch.save({"dataframe": df, "all_roles": LABELS}, save_path)

    return df


def extract_paragraph_containing_entity(document, start_offset, end_offset):
    """
    Extract the paragraph containing the entity based on its character offsets.
    """
    paragraphs = document.split("\n\n")
    current_position = 0
    for paragraph in paragraphs:
        start = current_position
        end = start + len(paragraph)
        if start <= start_offset < end or start < end_offset <= end:
            return paragraph
        current_position = end + 2
    return ""

def replace_entity_with_token(text, start_offset, end_offset, token="<ENTITY>"):
    """
    Replace an entity in the text with a placeholder token based on offsets.
    """
    return text[:start_offset]  + "<ENTITY_START>" +  text[start_offset:end_offset] +  "<ENTITY_END>"+ text[end_offset:]

def load_annotations_no_tokenization(annotation_file, documents_dir, is_test = False):
    """
    Load annotations, preprocess the text without tokenization, and return a DataFrame.

    Args:
        annotation_file (str): Path to the annotation file.
        documents_dir (str): Directory containing the raw documents.

    Returns:
        pd.DataFrame: A DataFrame containing preprocessed paragraphs and roles.
    """
    with open(annotation_file, "r", encoding="utf-8") as file:
        annotation_lines = file.readlines()

    structured_data = []


    all_roles = set()

    for line in annotation_lines:
        roles = line.strip().split("\t")[5:]
        all_roles.update(roles)

    for line in annotation_lines:
        parts = line.strip().split("\t")
        file_id = parts[0]
        entity = parts[1]
        start = int(parts[2])
        end = int(parts[3])
        roles = parts[4:] if not is_test else []

        document_path = os.path.join(documents_dir, file_id)
        if os.path.exists(document_path):
            with open(document_path, "r", encoding="utf-8") as doc_file:
                document = doc_file.read()
        else:
            print(f"Warning: File {file_id} not found.")
            document = ""

        paragraph, paragraph_start = extract_paragraph_containing_entity_with_offset(document, start, end)

        relative_start = start - paragraph_start
        relative_end = end - paragraph_start

        paragraph_with_entity = replace_entity_with_token(paragraph, relative_start, relative_end+1, "<ENTITY>")
        
        entity = {
            "File": file_id,
            "Original Entity": entity,
            "Processed Paragraph": paragraph_with_entity,
            "Start": start,
            "End": end
        }

        print(roles)
        if not is_test:
            label_vector = [0] * len(LABELS)
            for role in roles:
                if role in LABELS:
                    role_index = list(LABELS).index(role)
                    label_vector[role_index] = 1

            print(MAIN_ROLES.index(parts[4]))
            entity["main_role"] = roles[0]
            entity["fine_grained_roles"] = roles[1:]
            entity["multi_labels"] = label_vector
            entity["main_role_label"] = MAIN_ROLES.index(parts[4])
        structured_data.append(entity)

    return pd.DataFrame(structured_data)




def split_dataset_with_stratification(df, n_splits=10, random_state=42):
    """
    Split the dataset into training and validation sets using multi-label stratification.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        all_roles (list): List of all fine-grained roles.
        n_splits (int): Number of splits for stratification (default: 5).
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: Train and validation DataFrames.
    """
    labels = np.array(df["labels"].tolist())

    mss = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_index, val_index in mss.split(np.zeros(len(df)), labels):
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]
        break 

    return train_df, val_df


def extract_paragraph_containing_entity_with_offset(document, start_offset, end_offset):
    """
    Extract the paragraph containing the entity based on its character offsets,
    and return the paragraph along with its starting offset in the document.

    Args:
        document (str): The full document text.
        start_offset (int): The start offset of the entity in the document.
        end_offset (int): The end offset of the entity in the document.

    Returns:
        tuple: (paragraph text, paragraph starting offset in the document)
    """
    paragraphs = document.split("\n\n")
    current_position = 0
    for paragraph in paragraphs:
        paragraph_start = current_position
        paragraph_end = paragraph_start + len(paragraph)


        if paragraph_start <= start_offset < paragraph_end or paragraph_start < end_offset <= paragraph_end:
            return paragraph, paragraph_start

        current_position = paragraph_end + 2  

    return "", -1 