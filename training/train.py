from tqdm import tqdm
import torch
import torch.nn.functional as F


def train_model(model, dataloader, optimizer, device, writer, epoch, main_loss_fn, fine_loss_fn, patience,alpha=0.5):
    """
    Train the model for one epoch and log metrics to TensorBoard.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to train on (CPU or GPU).
        writer (SummaryWriter): TensorBoard writer instance.
        epoch (int): Current epoch number.
        main_loss_fn (nn.Module): Loss function for main role classification (e.g., CrossEntropyLoss).
        fine_loss_fn (nn.Module): Loss function for fine-grained role classification (e.g., BCEWithLogitsLoss).
        alpha (float): Weight for combining main role and fine-grained role losses.

    Returns:
        float: Average combined loss over the epoch.
    """
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_fine_loss = 0
    

    for step, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_main = batch["main_role_label"].to(device)  
        labels_fine = batch["labels"].to(device)  

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)

        main_logits = outputs["main_logits"]  # Main role logits
        fine_logits = outputs["fine_logits"]  # Fine-grained role logits

        main_loss = main_loss_fn(main_logits, labels_main)
        fine_loss = fine_loss_fn(fine_logits, labels_fine)

        loss = alpha * main_loss + (1 - alpha) * fine_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_fine_loss += fine_loss.item()


        if step % 10 == 0:
            writer.add_scalar("Train/Main Loss", main_loss.item(), epoch * len(dataloader) + step)
            writer.add_scalar("Train/Fine-Grained Loss", fine_loss.item(), epoch * len(dataloader) + step)
            writer.add_scalar("Train/Combined Loss", loss.item(), epoch * len(dataloader) + step)



   
    avg_loss = total_loss / len(dataloader)
    avg_main_loss = total_main_loss / len(dataloader)
    avg_fine_loss = total_fine_loss / len(dataloader)

    writer.add_scalar("Train/Average Main Loss", avg_main_loss, epoch)
    writer.add_scalar("Train/Average Fine-Grained Loss", avg_fine_loss, epoch)
    writer.add_scalar("Train/Average Combined Loss", avg_loss, epoch)

    current_lr = optimizer.param_groups[0]["lr"]
    writer.add_scalar("Train/Learning Rate", current_lr, epoch)

    return avg_loss


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True