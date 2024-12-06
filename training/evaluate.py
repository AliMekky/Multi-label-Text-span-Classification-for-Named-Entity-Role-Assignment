from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import torch

def evaluate_model(model, dataloader, device, writer, epoch, ALPHA, threshold=0.3):
    """
    Evaluate the model and log metrics to TensorBoard.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the validation data.
        device (torch.device): Device to evaluate on (CPU or GPU).
        writer (SummaryWriter): TensorBoard writer instance.
        epoch (int): Current epoch number.
        threshold (float): Threshold for multi-label classification.

    Returns:
        dict: Metrics including loss, precision, recall, and F1-score for both tasks.
    """
    model.eval()
    total_main_loss = 0
    total_fine_loss = 0
    all_main_preds, all_main_labels = [], []
    all_fine_preds, all_fine_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_main = batch["main_role_label"].to(device)
            labels_fine = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            main_logits = outputs["main_logits"]
            fine_logits = outputs["fine_logits"]

            main_preds = torch.argmax(main_logits, dim=-1)  # Multi-class predictions
            all_main_preds.append(main_preds.cpu())
            all_main_labels.append(labels_main.cpu())

            fine_preds = torch.sigmoid(fine_logits) > threshold  # Multi-label predictions
            all_fine_preds.append(fine_preds.cpu())
            all_fine_labels.append(labels_fine.cpu())

            main_loss_fn = torch.nn.CrossEntropyLoss()
            fine_loss_fn = torch.nn.BCEWithLogitsLoss()
            main_loss = main_loss_fn(main_logits, labels_main)
            fine_loss = fine_loss_fn(fine_logits, labels_fine)

            total_main_loss += main_loss.item()
            total_fine_loss += fine_loss.item()
    alpha = ALPHA
    loss = alpha * main_loss + (1 - alpha) * fine_loss
    all_main_preds = torch.cat(all_main_preds, dim=0).numpy()
    all_main_labels = torch.cat(all_main_labels, dim=0).numpy()
    all_fine_preds = torch.cat(all_fine_preds, dim=0).numpy()
    all_fine_labels = torch.cat(all_fine_labels, dim=0).numpy()

    avg_main_loss = total_main_loss / len(dataloader)
    avg_fine_loss = total_fine_loss / len(dataloader)
    avg_val_loss = loss/len(dataloader)

    main_precision = precision_score(all_main_labels, all_main_preds, average="macro", zero_division=0)
    main_recall = recall_score(all_main_labels, all_main_preds, average="macro", zero_division=0)
    main_f1 = f1_score(all_main_labels, all_main_preds, average="macro", zero_division=0)

    fine_precision = precision_score(all_fine_labels, all_fine_preds, average="macro", zero_division=0)
    fine_recall = recall_score(all_fine_labels, all_fine_preds, average="macro", zero_division=0)
    fine_f1 = f1_score(all_fine_labels, all_fine_preds, average="macro", zero_division=0)

    writer.add_scalar("Validation/Main Loss", avg_main_loss, epoch)
    writer.add_scalar("Validation/Fine Loss", avg_fine_loss, epoch)
    writer.add_scalar("Validation/Main Precision", main_precision, epoch)
    writer.add_scalar("Validation/Main Recall", main_recall, epoch)
    writer.add_scalar("Validation/Main F1 Score", main_f1, epoch)
    writer.add_scalar("Validation/Fine Precision", fine_precision, epoch)
    writer.add_scalar("Validation/Fine Recall", fine_recall, epoch)
    writer.add_scalar("Validation/Fine F1 Score", fine_f1, epoch)
    writer.add_scalar("Validation/avg_combined_loss_val", avg_val_loss, epoch)

    print(f"Validation Main Loss: {avg_main_loss}")
    print(f"Validation Main Precision: {main_precision}")
    print(f"Validation Main Recall: {main_recall}")
    print(f"Validation Main F1 Score: {main_f1}")
    print(f"Validation Fine Loss: {avg_fine_loss}")
    print(f"Validation Fine Precision: {fine_precision}")
    print(f"Validation Fine Recall: {fine_recall}")
    print(f"Validation Fine F1 Score: {fine_f1}")

    return {
        "main_loss": avg_main_loss,
        "fine_loss": avg_fine_loss,
        "main_precision": main_precision,
        "main_recall": main_recall,
        "main_f1": main_f1,
        "fine_precision": fine_precision,
        "fine_recall": fine_recall,
        "fine_f1": fine_f1,
        "avg_combined_loss_val": avg_val_loss
    }