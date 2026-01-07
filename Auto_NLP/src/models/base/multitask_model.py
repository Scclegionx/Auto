import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_intents: int,
        num_entity_labels: int,
        num_commands: int,
        dropout: float = 0.1,
        use_mean_pooling: bool = True,
        use_safetensors: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name,
            use_safetensors=use_safetensors,
            trust_remote_code=False,
        )
        self.hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.use_mean_pooling = use_mean_pooling

        self.intent_classifier = nn.Linear(self.hidden_size, num_intents)
        self.command_classifier = nn.Linear(self.hidden_size, num_commands)
        self.entity_classifier = nn.Linear(self.hidden_size, num_entity_labels)

    def _pool_sequence_output(
        self, sequence_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if not self.use_mean_pooling:
            return sequence_output[:, 0]

        mask = attention_mask.unsqueeze(-1)
        masked_output = sequence_output * mask
        summed = masked_output.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        return summed / lengths

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output: torch.Tensor = encoder_outputs.last_hidden_state
        pooled_output = self._pool_sequence_output(sequence_output, attention_mask)
        pooled_output = self.dropout(pooled_output)
        sequence_output = self.dropout(sequence_output)

        intent_logits = self.intent_classifier(pooled_output)
        command_logits = self.command_classifier(pooled_output)
        entity_logits = self.entity_classifier(sequence_output)

        return {
            "intent_logits": intent_logits,
            "command_logits": command_logits,
            "entity_logits": entity_logits,
        }

    @torch.no_grad()
    def predict(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        outputs = self.forward(input_ids, attention_mask)
        outputs["intent_probabilities"] = torch.softmax(outputs["intent_logits"], dim=-1)
        outputs["command_probabilities"] = torch.softmax(outputs["command_logits"], dim=-1)
        outputs["entity_probabilities"] = torch.softmax(outputs["entity_logits"], dim=-1)
        return outputs
