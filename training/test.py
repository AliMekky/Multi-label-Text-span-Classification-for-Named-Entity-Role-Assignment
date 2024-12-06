import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from models.model import TwoStepClassificationModel  
from data.preprocessing import load_and_tokenize_data
from configs.config import FINE_TO_MAIN_ROLE, LABELS, MAIN_ROLES, MODEL_NAME, MODEL_PATH, THRESHOLD, MAX_LENGTH, OUTPUT_FILE, ANNOTATION_FILE_TEST, DOCUMENTS_DIR_TEST, BATCH_SIZE, THRESHOLD


def infer_and_save_results(
    model_path, model_name, test_dataloader, fine_roles, main_roles, output_file, tokenizer, threshold=0.05
):
    """
    Perform inference on a test set using a DataLoader and save the results.

    Args:
        model_path (str): Path to the saved model weights.
        model_name (str): Hugging Face model name (e.g., "microsoft/deberta-v3-base").
        test_dataloader (DataLoader): DataLoader for the test set.
        fine_roles (list): List of all fine-grained roles (label names).
        main_roles (list): List of all main roles.
        output_file (str): File path to save the inference results.
        threshold (float): Minimum probability to include fine-grained roles.

    Returns:
        None
    """
    model = TwoStepClassificationModel(model_name, len(main_roles), len(fine_roles), tokenizer)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            file_ids = batch["File"]
            entities = batch["Original Entity"]
            start_offsets = batch["Start"]
            end_offsets = batch["End"]

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            main_logits, fine_logits = logits["main_logits"], logits["fine_logits"]
            main_probs = torch.softmax(main_logits, dim=-1).cpu().numpy()
            fine_probs = torch.sigmoid(fine_logits).cpu().numpy()

            for idx, (main_prob, fine_prob) in enumerate(zip(main_probs, fine_probs)):
                predicted_main_role_idx = main_prob.argmax()
                predicted_main_role = main_roles[predicted_main_role_idx]

                fine_grained_predictions = {
                    role: prob
                    for role, prob in zip(fine_roles, fine_prob)
                    if prob >= threshold and FINE_TO_MAIN_ROLE[role] == predicted_main_role
                }

                if not fine_grained_predictions:
                    max_role = fine_roles[fine_prob.argmax()]
                    fine_grained_predictions = {max_role: fine_prob.max()}

                sorted_fine_grained_roles = fine_grained_predictions.keys()

                results.append({
                    "File": file_ids[idx],
                    "Entity": entities[idx],
                    "Start Offset": start_offsets[idx].item(),
                    "End Offset": end_offsets[idx].item(),
                    "Main Role": predicted_main_role,
                    "Fine-Grained Roles": "\t".join(sorted_fine_grained_roles) 
                })

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(
                f"{result['File']}\t{result['Entity']}\t{result['Start Offset']}\t"
                f"{result['End Offset']}\t{result['Main Role']}\t{result['Fine-Grained Roles']}\n"
            )


if __name__ == "__main__":
    from data.dataset import RoleDataset

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    special_tokens = {"additional_special_tokens": ["<ENTITY_START>", "<ENTITY_END>"]}
    tokenizer.add_special_tokens(special_tokens) 
    test_df = load_and_tokenize_data(ANNOTATION_FILE_TEST, DOCUMENTS_DIR_TEST, tokenizer, MAX_LENGTH, is_test=True)
    test_dataset = RoleDataset(test_df, is_test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    infer_and_save_results(MODEL_PATH, MODEL_NAME, test_dataloader, LABELS, MAIN_ROLES, OUTPUT_FILE, tokenizer, THRESHOLD)
