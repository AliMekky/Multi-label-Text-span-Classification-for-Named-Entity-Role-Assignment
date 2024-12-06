from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from data.dataset import RoleDataset
from data.preprocessing import load_and_tokenize_data, split_dataset_with_stratification
from models.model import TwoStepClassificationModel, FocalLoss
from training.train import train_model, EarlyStopping
from training.evaluate import evaluate_model
from transformers import get_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from configs.config import MODEL_NAME, MODEL_PATH, ANNOTATION_FILE, DOCUMENTS_DIR, LOG_DIR, MAX_LENGTH, BATCH_SIZE, LEARNING_RATE, EPOCHS, MAIN_ROLES, MODEL_PATH, LABELS, ALPHA, PATIENCE


writer = SummaryWriter(log_dir=LOG_DIR)



# Paths
model_name = MODEL_NAME
annotation_file = ANNOTATION_FILE
documents_dir = DOCUMENTS_DIR
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Add custom tokens for <ENTITY_START> and <ENTITY_END>
special_tokens = {"additional_special_tokens": ["<ENTITY_START>", "<ENTITY_END>"]}
tokenizer.add_special_tokens(special_tokens)  # Update tokenizer with special tokens

# Constants
NUM_MAIN_ROLES = 3  # Protagonist, Antagonist, Innocent
NUM_FINE_GRAINED_ROLES = 22  # Total number of fine-grained roles
MAX_LENGTH = MAX_LENGTH  # Max sequence length
BATCH_SIZE = BATCH_SIZE  # Batch size
LEARNING_RATE = LEARNING_RATE  # Learning rate
NUM_EPOCHS = EPOCHS  # Number of epochs





df = load_and_tokenize_data(annotation_file, documents_dir, tokenizer, max_length=MAX_LENGTH)

train_dataset = RoleDataset(df)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = TwoStepClassificationModel(
    model_name=model_name,
    num_main_roles=NUM_MAIN_ROLES,
    num_fine_grained_roles=NUM_FINE_GRAINED_ROLES,
    tokenizer = tokenizer
)


# model = TwoStepClassificationModel(model_name, num_main_roles=len(MAIN_ROLES), num_fine_grained_roles=len(LABELS))
# model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
num_training_steps = len(train_dataloader) * NUM_EPOCHS
scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

class_counts = torch.tensor([103, 264, 47], dtype=torch.float)
class_weights = 1.0 / class_counts
normalized_weights = class_weights / class_weights.sum()  # Normalize weights
normalized_weights = normalized_weights.to(device)

# Loss functions
main_loss_fn = torch.nn.CrossEntropyLoss(weight = normalized_weights)
fine_loss_fn = FocalLoss()
# fine_loss_fn = torch.nn.BCEWithLogitsLoss()



best_combined_loss = float('inf')
best_model_path = "best_model.pth"
early_stopping = EarlyStopping(patience=PATIENCE)


for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

    
    avg_loss = train_model(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        device=device,
        writer=writer,
        epoch=epoch,
        main_loss_fn=main_loss_fn,
        fine_loss_fn=fine_loss_fn,
        patience= PATIENCE,
        alpha=ALPHA # Weight for main role vs fine-grained loss
    )
    print(f"Training Loss: {avg_loss}")
    scheduler.step()

    # Validate the model
    # val_metrics = evaluate_model(model, val_dataloader, device, writer, epoch, ALPHA)
    # print(f"Validation Loss: {val_metrics['avg_combined_loss_val']}")

    # Save the best model
    # if val_metrics['avg_combined_loss_val'] < best_combined_loss:
    #     best_combined_loss = val_metrics['avg_combined_loss_val']
    #     torch.save(model.state_dict(), best_model_path)
    #     print(f"New best model saved with validation loss: {best_combined_loss}")
    if avg_loss < best_combined_loss:
        best_combined_loss = avg_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with validation loss: {best_combined_loss}")
    # early_stopping(avg_loss)
    # if early_stopping.early_stop:
    #     print("Early stopping triggered.")
    #     break

writer.close()



"""
exp_1:
- description
    fine-tune roberta on multilabel fine-grained roles and get the main role from the predicted fine-grained labels.
- results:
    below the baseline
"""

"""
exp_2:
- description
    update the architecture of the model, by applying a mutliclass classification head to predict the main roles then take these outputs concatenated with the CLS token as an input to multilabel classification head to predict
    the fine-grained labels
- results:
    EMR = 0.13
"""

"""
exp_3:
- description
    update the loss functions to be weighted CE for main roles and Focal loss for the fine-grained loss. also if none of the fine-grained exceeded the threshold take the maximum 
- results:
    EMR = 0.18680, micro P = 0.23080, micro R = 0.36000, micro F1 = 0.28120, Accuracy for mian role = 0.76920
"""

"""
exp_4:
- description
    use XLM roberta and train for all languages
- results:
    EMR = 0.12 (also all the metrics were very bad)
"""

"""
exp_5:
- description
    use deberta
- results:
    below the baseline
"""

"""
exp_6:
- description
    use roberta-large
- results:
    below the baseline
"""

"""
exp_7:
- description
    apply the classification on the mean of the entity tokens not the CLS token 
- results:
    	0.12090	0.17890	0.17000	0.17440	0.74730
""" 

