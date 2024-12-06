import torch
from transformers import AutoTokenizer
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from models.model import TwoStepClassificationModel 
from configs.config import FINE_TO_MAIN_ROLE, LABELS, MAIN_ROLES, MODEL_PATH, MODEL_NAME, THRESHOLD, MAX_LENGTH


def infer_text(model_path, model_name, text, fine_roles, main_roles, max_length=512, threshold=0.05):
    """
    Perform inference on a given text using the fine-tuned multi-role classification model.

    Args:
        model_path (str): Path to the saved model weights.
        model_name (str): Hugging Face model name (e.g., "microsoft/deberta-v3-base").
        text (str): The input text to classify.
        fine_roles (list): List of all fine-grained roles (label names).
        main_roles (list): List of all main roles.
        max_length (int): Maximum sequence length for tokenization.
        threshold (float): Minimum probability to include fine-grained roles.

    Returns:
        dict: Filtered fine-grained roles and probabilities corresponding to the predicted main role.
        str: The predicted main role.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TwoStepClassificationModel(model_name, num_main_roles=len(main_roles), num_fine_grained_roles=len(fine_roles))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

 
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    
    inputs = {key: value.to(device) for key, value in inputs.items()}


    with torch.no_grad():
        results= model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        main_logits, fine_logits  = results["main_logits"], results["fine_logits"]
    

    main_probs = torch.softmax(main_logits, dim=-1).squeeze(0).cpu().numpy()
    predicted_main_role_idx = main_probs.argmax()
    predicted_main_role = main_roles[predicted_main_role_idx]


    fine_probs = torch.sigmoid(fine_logits).squeeze(0).cpu().numpy()
    print(fine_probs)

    filtered_fine_grained_roles = {
        role: prob
        for role, prob in zip(fine_roles, fine_probs)
        if prob >= threshold
    }

    return filtered_fine_grained_roles, predicted_main_role


if __name__ == "__main__":
    INPUT_TEXT = "But it gets even worse, as <ENTITY_START>Robert F. Kennedy<ENTITY_END> explained in New York City. The bioweapon is targeted to take out white people and spare other races."

    filtered_fine_grained_roles, predicted_main_role = infer_text(
        MODEL_PATH, MODEL_NAME, INPUT_TEXT, LABELS, MAIN_ROLES, threshold=THRESHOLD
    )

    print("\nFiltered Fine-Grained Role Predictions:")
    for role, prob in filtered_fine_grained_roles.items():
        print(f"{role}: {prob:.4f}")

    print("\nPredicted Main Role:")
    print(predicted_main_role)
