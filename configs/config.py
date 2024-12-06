# Configuration file for constants and settings used across the project

import os

# Paths
MODEL_PATH = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/code/best_model.pth"  
# MODEL_NAME = "roberta-base"  
MODEL_NAME = "FacebookAI/roberta-large"
TRAIN_DATA_PATH = "/path/to/train_data.pt"  
TEST_DATA_PATH = "/path/to/test_data.pt"  


# ## English
ANNOTATION_FILE = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/EN/subtask-1-annotations.txt" 
DOCUMENTS_DIR = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/EN/raw-documents"  
## BG
# ANNOTATION_FILE = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/BG/subtask-1-annotations.txt" 
# DOCUMENTS_DIR = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/BG/raw-documents"  
# ## HI
# ANNOTATION_FILE = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/HI/subtask-1-annotations.txt" 
# DOCUMENTS_DIR = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/HI/raw-documents"  
# ## PT
# ANNOTATION_FILE = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/PT/subtask-1-annotations.txt" 
# DOCUMENTS_DIR = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/PT/raw-documents"  

# ## English
ANNOTATION_FILE_TEST = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/EN/subtask-1-entity-mentions.txt" 
DOCUMENTS_DIR_TEST = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/EN/subtask-1-documents"  
## BG
# ANNOTATION_FILE_TEST = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/BG/subtask-1-entity-mentions.txt" 
# DOCUMENTS_DIR_TEST = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/BG/subtask-1-documents" 
# ## HI
# ANNOTATION_FILE_TEST = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/HI/subtask-1-entity-mentions.txt" 
# DOCUMENTS_DIR_TEST = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/HI/subtask-1-documents" 
# ## PT
# ANNOTATION_FILE_TEST = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/PT/subtask-1-entity-mentions.txt" 
# DOCUMENTS_DIR_TEST = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/PT/subtask-1-documents" 

OUTPUT_FILE = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/code/training/output.txt"  
LOG_DIR = "/home/ali.mekky/Documents/NLP/Assignment_2/SemEval2024/code/results/logs"  

# Model and Tokenizer Settings
MAX_LENGTH = 512  
BATCH_SIZE = 8  
LEARNING_RATE = 5e-5  
EPOCHS = 75
THRESHOLD = 0.5
ALPHA = 0.2
PATIENCE = 5

LABELS = [
    # Protagonist Roles
    "Martyr", "Peacemaker",  "Guardian", "Rebel", "Underdog", "Virtuous",

    # Antagonist Roles
    "Incompetent", "Instigator",  "Saboteur", "Conspirator", "Foreign Adversary", "Tyrant", "Corrupt", "Traitor",
    "Spy", "Terrorist", "Deceiver", "Bigot",

    # Innocent Roles
    "Forgotten", "Exploited", "Victim", "Scapegoat"
]


FINE_TO_MAIN_ROLE = {
    # Protagonist Roles
    "Guardian": "Protagonist",
    "Martyr": "Protagonist",
    "Peacemaker": "Protagonist",
    "Rebel": "Protagonist",
    "Underdog": "Protagonist",
    "Virtuous": "Protagonist",

    # Antagonist Roles
    "Instigator": "Antagonist",
    "Conspirator": "Antagonist",
    "Tyrant": "Antagonist",
    "Foreign Adversary": "Antagonist",
    "Traitor": "Antagonist",
    "Spy": "Antagonist",
    "Saboteur": "Antagonist",
    "Corrupt": "Antagonist",
    "Incompetent": "Antagonist",
    "Terrorist": "Antagonist",
    "Deceiver": "Antagonist",
    "Bigot": "Antagonist",

    # Innocent Roles
    "Forgotten": "Innocent",
    "Exploited": "Innocent",
    "Victim": "Innocent",
    "Scapegoat": "Innocent"
}


MAIN_ROLES = ["Protagonist", "Antagonist", "Innocent"]
