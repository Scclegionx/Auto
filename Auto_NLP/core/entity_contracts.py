"""
Entity Contracts
"""

# Entity whitelist per intent (only these entities are allowed)
ENTITY_WHITELIST = {
    "send-mess": {
        "required": ["MESSAGE", "RECEIVER"],
        "optional": ["PLATFORM"],
        "description": "Send message requires MESSAGE and RECEIVER, optionally PLATFORM"
    },
    "call": {
        "required": ["RECEIVER"],
        "optional": ["CONTACT_NAME", "PHONE"],
        "description": "Call requires RECEIVER (name or phone)"
    },
    "make-video-call": {
        "required": ["RECEIVER"],
        "optional": ["CONTACT_NAME", "PHONE", "PLATFORM"],
        "description": "Video call requires RECEIVER, optionally PLATFORM"
    },
    "set-alarm": {
        "required": ["TIME"],
        "optional": ["DATE", "TIMESTAMP", "DAYS_OF_WEEK", "FREQUENCY", "REMINDER_CONTENT", "RECURRENCE"],
        "description": "Alarm requires TIME, optionally DATE/TIMESTAMP/FREQUENCY/RECURRENCE/REMINDER_CONTENT"
    },
    "control-device": {
        "required": ["ACTION", "DEVICE"],
        "optional": [],
        "description": "Device control requires ACTION (ON/OFF) and DEVICE (flash/wifi/bluetooth/volume/brightness/data)"
    },
    "search-internet": {
        "required": ["QUERY"],
        "optional": ["KEYWORD", "LOCATION"],
        "description": "Internet search requires QUERY"
    },
    "search-youtube": {
        "required": ["QUERY"],
        "optional": ["YT_QUERY", "PLATFORM"],
        "description": "YouTube search requires QUERY"
    },
    "get-info": {
        "required": ["QUERY"],
        "optional": ["LOCATION", "TIME"],
        "description": "Get info (weather, time) requires QUERY"
    },
    "add-contacts": {
        "required": ["CONTACT_NAME"],
        "optional": ["PHONE", "PHONE_NUMBER"],
        "description": "Add contact requires CONTACT_NAME, optionally PHONE"
    },
    "open-cam": {
        "required": ["ACTION"],
        "optional": ["CAMERA_TYPE", "MODE"],
        "description": "Open camera requires ACTION"
    },
    "unknown": {
        "required": [],
        "optional": [],
        "description": "Unknown intent has no entities"
    }
}


def get_allowed_entities(intent: str) -> set:
    """Get set of allowed entities for an intent"""
    if intent not in ENTITY_WHITELIST:
        return set()
    
    contract = ENTITY_WHITELIST[intent]
    return set(contract["required"] + contract["optional"])


def get_required_entities(intent: str) -> set:
    """Get set of required entities for an intent"""
    if intent not in ENTITY_WHITELIST:
        return set()
    
    return set(ENTITY_WHITELIST[intent]["required"])


def filter_entities(intent: str, entities: dict) -> dict:

    allowed = get_allowed_entities(intent)
    if not allowed:
        return {}
    
    return {k: v for k, v in entities.items() if k in allowed}


def validate_entities(intent: str, entities: dict) -> tuple[bool, list[str]]:
    required = get_required_entities(intent)
    if not required:
        return True, []
    
    missing = [e for e in required if e not in entities or not entities[e]]
    return len(missing) == 0, missing


def calculate_entity_clarity_score(intent: str, entities: dict) -> float:
    score = 0.0
    
    # Check required entities (60% of score)
    required = get_required_entities(intent)
    if required:
        present = sum(1 for e in required if e in entities and entities[e])
        score += 0.6 * (present / len(required))
    else:
        score += 0.6  # No requirements = full score for this part
    
    # Check for unwanted entities (40% of score)
    allowed = get_allowed_entities(intent)
    if allowed:
        unwanted = sum(1 for k in entities.keys() if k not in allowed)
        total_entities = len(entities)
        if total_entities > 0:
            clean_ratio = 1 - (unwanted / total_entities)
            score += 0.4 * clean_ratio
        else:
            score += 0.4
    else:
        # No whitelist defined = any entity is "unwanted"
        score += 0.4 if len(entities) == 0 else 0.0
    
    return min(1.0, score)

