import torch
import torch.nn as nn
from transformers import AutoModel

class DebertaForMultiLabelClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        """
        A multi-label classification model based on DeBERTa.

        Args:
            model_name (str): Hugging Face model name (e.g., 'microsoft/deberta-v3-base').
            num_labels (int): Number of fine-grained roles (multi-label classification output size).
        """
        super(DebertaForMultiLabelClassification, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)  
        self.hidden_size = self.encoder.config.hidden_size   
        self.classifier = nn.Linear(self.hidden_size, num_labels) 

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs (batch_size x seq_length).
            attention_mask (torch.Tensor): Attention mask (batch_size x seq_length).
            labels (torch.Tensor, optional): Multi-label binary targets (batch_size x num_labels).

        Returns:
            torch.Tensor: Logits (batch_size x num_labels).
            torch.Tensor (optional): BCE loss if labels are provided.
        """
        # Encoder output
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get the CLS token output (batch_size x hidden_size)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token is at index 0
        
        # Pass through the classifier (batch_size x num_labels)
        logits = self.classifier(cls_output)
        
        if labels is not None:
            # Calculate BCE loss for multi-label classification
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            return loss, logits
        
        return logits



class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in multi-label classification tasks.
    Args:
        alpha (float): Scaling factor for positive examples. Default is 1.
        gamma (float): Focusing parameter. Default is 2.
        reduction (str): Reduction method. Options are 'none', 'mean', or 'sum'. Default is 'mean'.
    """
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Compute binary cross-entropy loss
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(logits, targets)
        pt = torch.exp(-bce_loss)  # Probability of the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        # Apply reduction method
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
        
import torch
import torch.nn as nn
from transformers import AutoModel


import torch
import torch.nn as nn
from transformers import AutoModel


class TwoStepClassificationModel(nn.Module):
    def __init__(self, model_name, num_main_roles, num_fine_grained_roles, tokenizer):
        """
        Two-step classification model using CLS token, main role predictions, 
        and embeddings between entity start and end tags.

        Args:
            model_name (str): Hugging Face model name (e.g., "roberta-base").
            num_main_roles (int): Number of main roles (multi-class classification output size).
            num_fine_grained_roles (int): Number of fine-grained roles (multi-label classification output size).
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.resize_token_embeddings(len(tokenizer))  # Resize model embeddings to match tokenizer
        self.hidden_size = self.encoder.config.hidden_size

        # Main role classification head (multi-class)
        self.main_role_classifier = nn.Linear(self.hidden_size, num_main_roles)

        # Fine-grained classification head (multi-label)
        self.fine_grained_classifier = nn.Sequential(
            # nn.Linear(self.hidden_size * 2 + num_main_roles, 256),  # CLS + entity span + main_probs
            nn.Linear(self.hidden_size + num_main_roles, 256),  # CLS + entity span + main_probs
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_fine_grained_roles)
        )

    def extract_entity_embeddings(self, token_embeddings, input_ids, entity_start_id, entity_end_id):
        """
        Extract embeddings between entity start and end tokens.

        Args:
            token_embeddings (torch.Tensor): Token embeddings (batch_size x seq_length x hidden_size).
            input_ids (torch.Tensor): Input token IDs (batch_size x seq_length).
            entity_start_id (int): ID corresponding to <ENTITY_START>.
            entity_end_id (int): ID corresponding to <ENTITY_END>.

        Returns:
            torch.Tensor: Mean-pooled embeddings between entity start and end tokens (batch_size x hidden_size).
        """
        entity_embeddings = []
        for i, seq_ids in enumerate(input_ids):
            try:
                # Find the start and end indices for the entity
                start_idx = (seq_ids == entity_start_id).nonzero(as_tuple=True)[0].item()
                end_idx = (seq_ids == entity_end_id).nonzero(as_tuple=True)[0].item()

                # Extract embeddings between the start and end tokens
                entity_span = token_embeddings[i, start_idx + 1:end_idx, :]  # Exclude start and end tokens
                if entity_span.size(0) > 0:  # Avoid empty spans
                    mean_embedding = entity_span.mean(dim=0)  # Mean pool
                else:
                    mean_embedding = torch.zeros(self.hidden_size, device=token_embeddings.device)
            except Exception as e:
                print(f"Error processing sequence {i}: {e}")
                mean_embedding = torch.zeros(self.hidden_size, device=token_embeddings.device)

            entity_embeddings.append(mean_embedding)

        return torch.stack(entity_embeddings, dim=0)  # (batch_size x hidden_size)


    def forward(self, input_ids, attention_mask, labels_main=None, labels_fine=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs (batch_size x seq_length).
            attention_mask (torch.Tensor): Attention mask (batch_size x seq_length).
            entity_start_ids (list[int]): IDs corresponding to <ENTITY_START>.
            entity_end_ids (list[int]): IDs corresponding to <ENTITY_END>.
            labels_main (torch.Tensor, optional): Main role labels (batch_size).
            labels_fine (torch.Tensor, optional): Fine-grained role labels (batch_size x num_fine_grained_roles).

        Returns:
            dict: Dictionary containing logits and losses for both tasks.
        """

        entity_start_ids, entity_end_ids = [50265, 50266]
        # Encoder output
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # (batch_size x seq_length x hidden_size)
        cls_output = token_embeddings[:, 0, :]  # CLS token: (batch_size x hidden_size)

        # Extract entity embeddings
        # entity_embeddings = self.extract_entity_embeddings(token_embeddings, input_ids, entity_start_ids, entity_end_ids)

        # Step 1: Main role prediction
        # main_logits = self.main_role_classifier(torch.cat([cls_output, entity_embeddings], dim = -1))  # (batch_size x num_main_roles)
        main_logits = self.main_role_classifier(cls_output)  # (batch_size x num_main_roles)
        main_probs = torch.softmax(main_logits, dim=-1)  # Convert logits to probabilities

        # Step 2: Fine-grained role prediction
        # Concatenate CLS output, entity embeddings, and main role probabilities
        combined_input = torch.cat([cls_output, main_probs], dim=-1)  # (batch_size x hidden_size * 2 + num_main_roles)
        fine_logits = self.fine_grained_classifier(combined_input)  # (batch_size x num_fine_grained_roles)

        # Loss calculations
        losses = {}
        if labels_main is not None:
            main_loss_fn = nn.CrossEntropyLoss()
            losses["main_loss"] = main_loss_fn(main_logits, labels_main)

        if labels_fine is not None:
            fine_loss_fn = nn.BCEWithLogitsLoss()
            losses["fine_loss"] = fine_loss_fn(fine_logits, labels_fine)

        return {
            "main_logits": main_logits,
            "fine_logits": fine_logits,
            "losses": losses
        }
