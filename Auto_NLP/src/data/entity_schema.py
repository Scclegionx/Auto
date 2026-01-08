
from __future__ import annotations

from typing import Dict, Iterable, List


ENTITY_BASE_NAMES: List[str] = [
    "ACTION",
    "CAMERA_TYPE",
    "CONTACT_NAME",
    "DATE",
    "DEVICE",
    "FREQUENCY",
    "LEVEL",
    "LOCATION",
    "MESSAGE",
    "MODE",
    "PHONE",
    "PLATFORM",
    "QUERY",
    "RECEIVER",
    "REMINDER_CONTENT",
    "TIME",
]


def generate_entity_labels() -> List[str]:
    """Sinh danh sách nhãn BIO chuẩn theo ENTITY_BASE_NAMES."""
    labels: List[str] = ["O"]
    labels.extend(f"B-{name}" for name in ENTITY_BASE_NAMES)
    labels.extend(f"I-{name}" for name in ENTITY_BASE_NAMES)
    return labels


LEGACY_TO_CANONICAL: Dict[str, str] = {
    # Legacy field regrouping
    "TITLE": "REMINDER_CONTENT",
    "LABEL": "REMINDER_CONTENT",
    "CONTENT_TYPE": "QUERY",
    "MEDIA_TYPE": "QUERY",
    "PHONE_NUMBER": "PHONE",
    # Prefixed variants
    "B-TITLE": "B-REMINDER_CONTENT",
    "I-TITLE": "I-REMINDER_CONTENT",
    "B-LABEL": "B-REMINDER_CONTENT",
    "I-LABEL": "I-REMINDER_CONTENT",
    "B-CONTENT_TYPE": "B-QUERY",
    "I-CONTENT_TYPE": "I-QUERY",
    "B-MEDIA_TYPE": "B-QUERY",
    "I-MEDIA_TYPE": "I-QUERY",
    "B-PHONE_NUMBER": "B-PHONE",
    "I-PHONE_NUMBER": "I-PHONE",
}


def canonicalize_entity_label(label: str) -> str:
    """Đưa nhãn entity về dạng chuẩn (bao gồm cả nhãn legacy)."""
    if not label:
        return label
    return LEGACY_TO_CANONICAL.get(label, label)


def canonicalize_entity_dict(entity: Dict[str, str]) -> Dict[str, str]:
    """Trả về bản sao entity dict với label đã chuẩn hóa."""
    if not entity:
        return entity
    canonical = dict(entity)
    canonical["label"] = canonicalize_entity_label(canonical.get("label", ""))
    if "type" in canonical:
        canonical["type"] = canonicalize_entity_label(canonical.get("type", ""))
    return canonical


def canonicalize_label_sequence(labels: Iterable[str]) -> List[str]:
    """Chuẩn hóa một dãy nhãn (ví dụ chuỗi BIO)."""
    return [canonicalize_entity_label(label) for label in labels]


