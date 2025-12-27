from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class MultiTaskDataset(Dataset):
    """Dataset chuẩn cho multi-task learning (intent/entity/command)."""

    def __init__(self, samples: Iterable[Dict]) -> None:
        self.samples: List[Dict] = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        input_ids = torch.as_tensor(item["input_ids"], dtype=torch.long)
        attention_mask = torch.as_tensor(item["attention_mask"], dtype=torch.long)
        intent_labels = torch.as_tensor(item["intent_label"], dtype=torch.long)
        entity_labels = torch.as_tensor(item["entity_labels"], dtype=torch.long)
        command_labels = torch.as_tensor(item["command_label"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "intent_labels": intent_labels,
            "entity_labels": entity_labels,
            "command_labels": command_labels,
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    sampler: Optional[Sampler[int]] = None,
) -> DataLoader:
    """Utility tạo DataLoader chuẩn (đã cố định chiều dài nên không cần collate đặc biệt)."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        pin_memory=True,
        num_workers=num_workers,
    )

