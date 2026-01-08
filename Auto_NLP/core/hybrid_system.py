import torch
import torch.nn as nn
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from transformers import AutoTokenizer, AutoModel
import logging
import re
import sys
import time
import unicodedata
import types
import re

current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

try:
    import torchvision  # noqa: F401
except ImportError:
    # Create mock torchvision module if not available (not required for this system)
    mock_tv = types.ModuleType("torchvision")
    transforms_mod = types.ModuleType("transforms")

    class _InterpolationMode:
        BILINEAR = "bilinear"
        NEAREST = "nearest"

    transforms_mod.InterpolationMode = _InterpolationMode  # type: ignore[attr-defined]
    mock_tv.transforms = transforms_mod  # type: ignore[attr-defined]

    sys.modules["torchvision"] = mock_tv
    sys.modules["torchvision.transforms"] = transforms_mod

try:
    from core.reasoning_engine import ReasoningEngine
    from core.model_loader import TrainedModelInference, load_trained_model
    from src.inference.engines.entity_extractor import EntityExtractor as SpecializedEntityExtractor
    from core.entity_contracts import filter_entities, validate_entities, calculate_entity_clarity_score
except ImportError as e:
    logging.error(f"Failed to import components: {e}")
    sys.exit(1)

class ModelFirstHybridSystem:
    """
    Model-First Hybrid System:
    - Trained model làm chính (primary prediction)
    - Reasoning engine làm phụ (validation, enhancement, fallback)
    """
    
    def __init__(self, model_path: str = "models/phobert_multitask"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        
        # Components
        self.trained_model = None
        self.model_inference: Optional[TrainedModelInference] = None
        self.tokenizer = None
        self.reasoning_engine = None
        self.label_mappings = None
        self.config = None
        
        # Status
        self.model_loaded = False
        self.reasoning_loaded = False
        
        # Performance tracking
        self.stats = {
            'total_predictions': 0,
            'model_predictions': 0,
            'reasoning_predictions': 0,
            'hybrid_predictions': 0,
            'fallback_predictions': 0,
            'avg_processing_time': 0.0,
            'confidence_scores': []
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self._initialize_components()
    
    @staticmethod
    def _resolve_alarm_datetime(text: str) -> Dict[str, str]:
        try:
            import datetime as _dt
            import re as _re
            from zoneinfo import ZoneInfo

            TZ = ZoneInfo("Asia/Bangkok")
            now = _dt.datetime.now(tz=TZ)
            text_l = (text or "").lower()
            text_norm = ModelFirstHybridSystem._normalize_text(text_l)

            def _iter_sources(raw: str) -> List[str]:
                sources = [raw]
                if text_norm and text_norm not in sources:
                    sources.append(text_norm)
                return sources

            def _parse_vi_time(raw: str):
                if not raw:
                    return (None, None)
                s = raw.strip().lower()
                m = _re.search(r"\b(\d{1,2})\s*(giờ|gio|h)?\s*rưỡi\b", s)
                if m:
                    return (int(m.group(1)), 30)
                m = _re.search(r"\b(\d{1,2})\s*(giờ|gio|h)?\s*kém\s*(\d{1,2})\b", s)
                if m:
                    h = int(m.group(1))
                    k = int(m.group(3))
                    h = (h - 1) if k > 0 else h
                    return (h, (60 - k) % 60)
                m = _re.search(r"\b(\d{1,2})\s*(?:giờ|gio|h)\s*(\d{1,2})\b", s)
                if m:
                    return (int(m.group(1)), int(m.group(2)))
                m = _re.search(r"\b(\d{1,2}):(\d{1,2})\b", s)
                if m:
                    return (int(m.group(1)), int(m.group(2)))
                m = _re.search(r"\b(\d{1,2})\s*(giờ|gio|h)\b", s)
                if m:
                    return (int(m.group(1)), 0)
                m = _re.fullmatch(r"\s*(\d{1,2})\s*", s)
                if m:
                    return (int(m.group(1)), 0)
                return (None, None)

            def _apply_period(h: int, mi: int):
                if h is None:
                    return (None, None)
                if h >= 13:
                    return (h, mi)
                has_sang = any(k in source for source in _iter_sources(text_l) for k in ["sáng", "sang"])
                has_trua = any(k in source for source in _iter_sources(text_l) for k in ["trưa", "giua trua", "nua trua"])
                has_chieu = any(k in source for source in _iter_sources(text_l) for k in ["chiều", "chieu"])
                has_toi = any(k in source for source in _iter_sources(text_l) for k in ["tối", "toi"])
                has_dem = any(k in source for source in _iter_sources(text_l) for k in ["đêm", "dem", "khuya", "nửa đêm", "nua dem"])

                if h == 12:
                    if has_trua:
                        return (12, mi)
                    if has_dem:
                        return (0, mi)
                    return (12, mi)

                if has_sang:
                    return (h, mi)
                if has_trua:
                    if 1 <= h <= 5:
                        return (h + 12, mi)
                    return (12 if h == 0 else h, mi)
                if has_chieu:
                    return (h + 12, mi)
                if has_toi:
                    hh = h + 12
                    if hh < 18:
                        hh = 18
                    return (hh, mi)
                if has_dem:
                    if 1 <= h <= 4:
                        return (h, mi)
                    if 10 <= h <= 11:
                        return (h + 12, mi)
                    if h == 9:
                        return (21, mi)
                    if h == 8:
                        return (20, mi)
                    return (h, mi)
                return (h, mi)

            h = mi = None
            for src in _iter_sources(text_l):
                h_candidate, mi_candidate = _parse_vi_time(src)
                if h_candidate is not None:
                    h, mi = h_candidate, mi_candidate
                    break
            if h is not None:
                h, mi = _apply_period(h, 0 if mi is None else mi)

            if h is not None:
                mi = 0 if mi is None else max(0, min(59, mi))
                h = max(0, min(23, h))
                time_str = f"{h:02d}:{mi:02d}"
            else:
                time_str = None

            date_obj = None
            sources = _iter_sources(text_l)
            if any("hôm nay" in src or "hom nay" in src for src in sources):
                date_obj = now.date()
            elif any(k in src for src in sources for k in ["ngày mai", "mai", "ngay mai"]):
                date_obj = (now + _dt.timedelta(days=1)).date()
            elif any(k in src for src in sources for k in ["ngày mốt", "mốt", "ngày kia", "ngay mot", "ngay kia"]):
                date_obj = (now + _dt.timedelta(days=2)).date()
            elif any("tuần sau" in src or "tuan sau" in src for src in sources):
                date_obj = (now + _dt.timedelta(days=7)).date()

            weekday_map = {
                "thứ hai": 0,
                "thứ ba": 1,
                "thứ tư": 2,
                "thứ năm": 3,
                "thứ sáu": 4,
                "thứ bảy": 5,
                "chủ nhật": 6,
            }
            for vn, wd in weekday_map.items():
                if any(vn in src or ModelFirstHybridSystem._normalize_text(vn) in src for src in sources):
                    delta = (wd - now.weekday()) % 7
                    if delta == 0:
                        delta = 7
                    if any("tuần sau" in src or "tuan sau" in src for src in sources):
                        delta += 7
                    date_obj = (now + _dt.timedelta(days=delta)).date()
                    break

            result = {}
            if time_str:
                result["TIME"] = time_str
            if time_str and date_obj is None:
                candidate_dt = _dt.datetime.combine(now.date(), _dt.time.fromisoformat(time_str), tzinfo=TZ)
                if candidate_dt <= now:
                    date_obj = now.date() + _dt.timedelta(days=1)
                else:
                    date_obj = now.date()
            if date_obj is not None:
                result["DATE"] = date_obj.isoformat()
            if result.get("TIME") and result.get("DATE"):
                hh, mm = [int(x) for x in result["TIME"].split(":")]
                d = _dt.date.fromisoformat(result["DATE"])
                ts = _dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=TZ)
                result["TIMESTAMP"] = ts.isoformat()

            return result
        except Exception:
            return {}

    @staticmethod
    def _normalize_text(text: Optional[str]) -> str:
        if not text:
            return ""
        normalized = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")

    @staticmethod
    def _extract_message_receiver(text: str) -> Dict[str, str]:
        """
        Heuristic parser send-mess, bám yêu cầu:
        - Case A/B/C: receiver qua cho/tới/đến; marker nội dung (rằng|là|nói|bảo|nhắn|gửi|nhắc|:) nếu có.
        - Platform chỉ khi có "qua|bằng|trên <platform_keyword>".
        - Heuristic cắt receiver ngắn (<=4 token) để tránh nuốt MESSAGE.
        """
        if not text:
            return {}

        original = text
        lower = text.lower()
        out: Dict[str, str] = {}

        # 1) Detect platform
        platform_keywords = ["zalo", "sms", "messenger", "facebook", "fb", "viber", "telegram", "whatsapp", "imessage"]
        platform = None
        m_plat = re.search(r"\b(qua|bằng|trên)\s+([a-zA-Z0-9_]+)", lower)
        if m_plat:
            cand = m_plat.group(2).lower()
            if cand in platform_keywords:
                platform = cand
        if platform:
            out["PLATFORM"] = platform

        # 2) Find marker positions
        marker_list = [" rằng ", " rằng là ", " là ", " bảo ", " nói ", " nhắn ", " gửi ", " nhắc ", ":"]
        marker_pos = []
        for mk in marker_list:
            pos = lower.find(mk)
            if pos != -1:
                marker_pos.append((pos, mk))
        marker_pos.sort(key=lambda x: x[0])
        first_marker = marker_pos[0] if marker_pos else None

        # 3) Find receiver via cho/tới/đến
        rec_match = re.search(r"\b(cho|tới|đến)\s+([^,:.;]+)", lower)
        receiver = None
        receiver_span = None
        if rec_match:
            span_start = rec_match.start(2)
            span_end = rec_match.end(2)
            candidate = original[span_start:span_end].strip()
            # Cắt receiver khi gặp marker/dấu câu trong candidate
            stop_tokens = [" rằng", " la", " là", " bảo", " nói", " nhắn", " gửi", " nhắc", ":", ",", "."]
            cand_lower = candidate.lower()
            cut_idx = None
            for st in stop_tokens:
                idx = cand_lower.find(st)
                if idx != -1:
                    cut_idx = idx
                    break
            if cut_idx is not None:
                candidate = candidate[:cut_idx].strip()
                span_end = span_start + cut_idx
            # Heuristic: limit receiver length to 4 tokens
            tokens = candidate.split()
            if len(tokens) > 4:
                candidate = " ".join(tokens[:3])
                span_end = span_start + len(" ".join(tokens[:3]))
            receiver = candidate.strip()
            receiver_span = (span_start, span_end)

        # 4) Message extraction based on marker and receiver ordering
        message = None
        if first_marker:
            m_pos, mk = first_marker
            mk_end = m_pos + len(mk)
            if receiver_span and m_pos > receiver_span[0]:
                # Marker sau receiver (Case A)
                message = original[mk_end:].strip()
            elif receiver_span and m_pos < receiver_span[0]:
                # Marker trước receiver (Case C)
                message = original[mk_end:receiver_span[0]].strip()
                receiver = original[receiver_span[0]:receiver_span[1]].strip() if receiver_span else receiver
            else:
                # Không có receiver, marker vẫn cắt message
                message = original[mk_end:].strip()
        else:
            # No marker: Case B, message là phần sau receiver (nếu có)
            if receiver_span:
                message = original[receiver_span[1]:].strip()

        # Clean receiver/message
        if receiver:
            out["RECEIVER"] = receiver.strip(" ,.:;")
        if message:
            out["MESSAGE"] = message.strip(" ,.:;")
        return out

    def _initialize_components(self):
        """Initialize all system components"""
        try:
            self._load_trained_model()
        except Exception as e:
            self.logger.error(f"Failed to load trained model: {e}")
            self.model_loaded = False
        
        try:
            self.reasoning_engine = ReasoningEngine()
            self.reasoning_loaded = True
        except Exception as e:
            self.logger.error(f"Failed to initialize reasoning engine: {e}")
            self.reasoning_loaded = False
        
        try:
            self.specialized_entity_extractor = SpecializedEntityExtractor()
            self.specialized_extractor_loaded = True
        except Exception as e:
            self.logger.error(f"Failed to initialize specialized entity extractor: {e}")
            self.specialized_extractor_loaded = False
    
    def _load_trained_model(self):
        """Load trained multi-task model (phobert_multitask) cho hybrid system."""
        try:
            device_str = str(self.device) if self.device.type == "cpu" else self.device.type
            self.model_inference = load_trained_model("phobert_multitask", device=device_str)
            self.enable_multi_task = True
            self.model_loaded = True
        except Exception as e:
            self.logger.error(f"Failed to load trained model: {e}")
            self.model_loaded = False
            raise
    
    def predict(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main prediction method - Model-first approach
        
        Flow:
        1. Trained model prediction (PRIMARY)
        2. Reasoning engine validation/enhancement (SECONDARY)
        3. Hybrid decision making
        4. Final result
        """
        start_time = time.time()
        self.stats['total_predictions'] += 1
        
        try:
            try:
                if self.reasoning_engine and hasattr(self.reasoning_engine, 'conversation_context'):
                    self.reasoning_engine.conversation_context.reset()
            except Exception:
                pass

            model_result = self._model_predict(text)
            reasoning_result = self._reasoning_validate(text, context, model_result)
            final_result = self._hybrid_decision(model_result, reasoning_result, text, context)
            final_result = self._postprocess_command_entities(text, final_result)
            processing_time = time.time() - start_time
            final_result.update({
                'timestamp': time.time(),
                'processing_time': processing_time,
                'input_text': text,
                'context': context or {},
                'model_result': model_result,
                'reasoning_result': reasoning_result
            })
            self._update_stats(final_result, processing_time)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            return self._fallback_prediction(text, context)

    def _postprocess_command_entities(self, text: str, result: Dict[str, Any]) -> Dict[str, Any]:
        try:
            text_l = (text or "").lower()
            command = result.get("command") or result.get("intent") or "unknown"
            entities = dict(result.get("entities") or {})

            if command == "control-device":
                # Use specialized extractor for mobile device control (flash, wifi, bluetooth, volume, brightness, data...)
                if hasattr(self, 'specialized_entity_extractor'):
                    try:
                        specialized_ents, conf = self.specialized_entity_extractor._extract_device_control_with_confidence(text, "control-device")
                        if conf >= 0.85:
                            entities.update(specialized_ents)
                    except Exception as e:
                        self.logger.warning(f"Specialized device extractor failed: {e}")
                
                if not entities.get("ACTION") or not entities.get("DEVICE"):
                    has_flash = any(w in text_l for w in ["đèn pin", "flash", "flashlight"])
                    has_wifi = any(w in text_l for w in ["wifi", "wi fi"])
                    has_bluetooth = any(w in text_l for w in ["bluetooth", "blutooth"])
                    has_volume = any(w in text_l for w in ["âm lượng", "tiếng", "volume"])
                    has_brightness = any(w in text_l for w in ["độ sáng", "sáng", "brightness"])
                    has_data = any(w in text_l for w in ["data", "3g", "4g", "5g", "dữ liệu"])
                    has_on = any(w in text_l for w in ["bật", "mở", "on"]) 
                    has_off = any(w in text_l for w in ["tắt", "đóng", "off"])
                    
                    if not entities.get("DEVICE"):
                        if has_flash:
                            entities["DEVICE"] = "flash"
                        elif has_wifi:
                            entities["DEVICE"] = "wifi"
                        elif has_bluetooth:
                            entities["DEVICE"] = "bluetooth"
                        elif has_volume:
                            entities["DEVICE"] = "volume"
                        elif has_brightness:
                            entities["DEVICE"] = "brightness"
                        elif has_data:
                            entities["DEVICE"] = "mobile_data"
                    
                    if not entities.get("ACTION"):
                        if has_on:
                            entities["ACTION"] = "ON"
                        elif has_off:
                            entities["ACTION"] = "OFF"
                
                # Clean up unrelated entities
                entities.pop("PLATFORM", None)
                entities.pop("MODE", None)
                entities.pop("VALUE", None)
                entities.pop("LEVEL", None)
                entities.pop("LOCATION", None)

                result["entities"] = entities

            if command == "set-alarm":
                entities.pop("PLATFORM", None)
                entities.pop("QUERY", None)
                entities.pop("YT_QUERY", None)
                
                if hasattr(self, 'specialized_entity_extractor'):
                    try:
                        specialized_ents, conf = self.specialized_entity_extractor._extract_alarm_time_date_with_confidence(text, "set-alarm")
                        if specialized_ents:
                            entities.update(specialized_ents)
                            result["entities"] = entities
                    except Exception as e:
                        self.logger.warning(f"Specialized alarm extractor failed: {e}")
                
                if not entities.get("REMINDER_CONTENT"):
                    entities["REMINDER_CONTENT"] = "Nhắc nhở"
                
                result["entities"] = entities
            
            if command == "add-contacts":
                for k in ["DEVICE", "ACTION", "MODE", "PLATFORM"]:
                    entities.pop(k, None)
                contacts_enriched = {}
                if hasattr(self, 'specialized_extractor_loaded') and self.specialized_extractor_loaded:
                    try:
                        contacts_enriched = self.specialized_entity_extractor._extract_contact_entities(text)  # noqa: SLF001
                    except Exception:
                        contacts_enriched = {}
                if contacts_enriched:
                    if contacts_enriched.get("CONTACT_NAME"):
                        entities["CONTACT_NAME"] = contacts_enriched["CONTACT_NAME"]
                    phone_candidate = contacts_enriched.get("PHONE") or contacts_enriched.get("PHONE_NUMBER")
                    if phone_candidate:
                        entities["PHONE"] = phone_candidate
                if not entities.get("PHONE") and entities.get("PHONE_NUMBER"):
                    entities["PHONE"] = entities["PHONE_NUMBER"]
                result["entities"] = entities

            if command == "call":
                entities.pop("PLATFORM", None)
                if entities.get("RECEIVER") and not entities.get("CONTACT_NAME") and not entities["RECEIVER"].isdigit():
                    entities["CONTACT_NAME"] = entities["RECEIVER"]
                if entities.get("CONTACT_NAME") and not entities.get("RECEIVER"):
                    entities["RECEIVER"] = entities["CONTACT_NAME"]
                result["entities"] = entities

            if command == "make-video-call":
                platform_value = entities.get("PLATFORM")
                if not platform_value:
                    platform_guess = ""
                    if hasattr(self, 'specialized_extractor_loaded') and self.specialized_extractor_loaded:
                        try:
                            platform_guess = self.specialized_entity_extractor.extract_platform(text)  # type: ignore[attr-defined]
                        except Exception:
                            platform_guess = ""
                    entities["PLATFORM"] = platform_guess or "zalo"
                if entities.get("RECEIVER") and not entities.get("CONTACT_NAME") and not entities["RECEIVER"].isdigit():
                    entities["CONTACT_NAME"] = entities["RECEIVER"]
                result["entities"] = entities

            if command == "send-mess":
                for k in ["QUERY", "DEVICE", "ACTION", "MODE", "LEVEL", "VALUE", "DATE", "DAYS_OF_WEEK", "TIMESTAMP", "REMINDER_CONTENT"]:
                    entities.pop(k, None)

                if hasattr(self, 'specialized_entity_extractor'):
                    try:
                        specialized_ents, conf = self.specialized_entity_extractor._extract_message_receiver_with_confidence(text, "send-mess")
                        if conf >= 0.8:
                            if specialized_ents.get("RECEIVER"):
                                entities["RECEIVER"] = specialized_ents["RECEIVER"]
                            if specialized_ents.get("MESSAGE"):
                                entities["MESSAGE"] = specialized_ents["MESSAGE"]
                            if specialized_ents.get("PLATFORM"):
                                entities["PLATFORM"] = specialized_ents["PLATFORM"]
                    except Exception as e:
                        self.logger.warning(f"Specialized message/receiver extractor failed: {e}")
                        parsed = self._extract_message_receiver(text)
                        if parsed.get("RECEIVER") and not entities.get("RECEIVER"):
                            entities["RECEIVER"] = parsed["RECEIVER"]
                        if parsed.get("MESSAGE"):
                            existing_msg = entities.get("MESSAGE", "")
                            if not existing_msg or len(parsed["MESSAGE"]) > len(existing_msg):
                                entities["MESSAGE"] = parsed["MESSAGE"]

                platform_in_text = any(
                    kw in text.lower()
                    for kw in ["zalo", "sms", "messenger", "facebook", "fb", "viber", "telegram"]
                )
                if not platform_in_text and "PLATFORM" in entities and entities["PLATFORM"] == "sms":
                    entities.pop("PLATFORM", None)

                result["entities"] = entities

            if command == "search-internet":
                # Remove control-device leakage
                for k in ["DEVICE", "ACTION", "MODE"]:
                    entities.pop(k, None)
                if any(w in text_l for w in ["youtube", "yt"]):
                    entities.pop("PLATFORM", None)
                if not entities.get("PLATFORM"):
                    entities["PLATFORM"] = "google"
                result["entities"] = entities

            if command == "search-youtube":
                entities["PLATFORM"] = "youtube"
                result["entities"] = entities

            if command == "send-mess":
                if not entities.get("PLATFORM"):
                    platform_guess = ""
                    if hasattr(self, 'specialized_extractor_loaded') and self.specialized_extractor_loaded:
                        try:
                            platform_guess = self.specialized_entity_extractor.extract_platform(text)  # type: ignore[attr-defined]
                        except Exception:
                            platform_guess = ""
                    entities["PLATFORM"] = platform_guess or "zalo"
                result["entities"] = entities

            filtered_entities = filter_entities(command, entities)
            result["entities"] = filtered_entities
            
            is_valid, missing = validate_entities(command, filtered_entities)
            if not is_valid:
                self.logger.warning(f"Command '{command}' missing required entities: {missing}")
            
            clarity_score = calculate_entity_clarity_score(command, filtered_entities)
            result["entity_clarity_score"] = round(clarity_score, 2)

            return result
        except Exception as e:
            self.logger.error(f"Error in _postprocess_command_entities: {e}")
            import traceback
            traceback.print_exc()
            return result
    
    def _model_predict(self, text: str) -> Dict[str, Any]:
        """Primary prediction using trained model"""
        if not self.model_loaded or self.model_inference is None:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "entities": {},
                "command": "unknown",
                "method": "model_error",
                "error": "Model not loaded"
            }
        
        try:
            self.stats['model_predictions'] += 1
            
            model_prediction = self.model_inference.predict(text)
            model_intent = model_prediction.get("intent", "unknown")
            model_command = model_prediction.get("command", model_intent)
            model_entities = model_prediction.get("entities", {}) or {}
            model_confidence = float(model_prediction.get("confidence", 0.0) or 0.0)

            heuristic_prediction = self._rule_based_model_prediction(text)
            heuristic_intent = heuristic_prediction.get("intent", "unknown")
            heuristic_confidence = float(heuristic_prediction.get("confidence", 0.0) or 0.0)
            heuristic_entities = heuristic_prediction.get("entities", {}) or {}

            intents_align = heuristic_intent == model_intent

            combined_confidence = model_confidence
            confidence_adjustments = []

            if intents_align and heuristic_confidence > 0:
                boost = min(0.25, 0.15 + 0.3 * heuristic_confidence)
                combined_confidence = min(1.0, combined_confidence + boost)
                confidence_adjustments.append({
                    "type": "keyword_boost",
                    "value": boost,
                    "keyword_confidence": heuristic_confidence
                })
            elif not intents_align and heuristic_confidence >= 0.5:
                penalty = min(0.25, 0.2 + 0.2 * heuristic_confidence)
                combined_confidence = max(0.05, combined_confidence - penalty)
                confidence_adjustments.append({
                    "type": "keyword_penalty",
                    "value": penalty,
                    "keyword_confidence": heuristic_confidence,
                    "alternate_intent": heuristic_intent
                })

            if combined_confidence < 0.4 and model_confidence >= 0.4:
                combined_confidence = (combined_confidence + model_confidence) / 2
                confidence_adjustments.append({
                    "type": "stability_floor",
                    "value": combined_confidence - model_confidence
                })

            merged_entities = dict(heuristic_entities)
            merged_entities.update(model_entities)

            return {
                "intent": model_intent,
                "confidence": combined_confidence,
                "entities": merged_entities,
                "command": model_command,
                "method": "trained_model",
                "model_type": model_prediction.get("model_type", "multi-task" if self.enable_multi_task else "single-task"),
                "confidence_sources": {
                    "model": model_confidence,
                    "keyword": heuristic_confidence,
                    "combined": combined_confidence,
                    "agreement": intents_align,
                    "adjustments": confidence_adjustments,
                },
                "auxiliary_intent": heuristic_intent,
                "auxiliary_entities": heuristic_entities,
            }
            
        except Exception as e:
            self.logger.error(f"Model prediction error: {e}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "entities": {},
                "command": "unknown",
                "method": "model_error",
                "error": str(e)
            }
    
    def _rule_based_model_prediction(self, text: str) -> Dict[str, Any]:
        """Rule-based prediction using patterns learned from trained model"""
        text_lower = text.lower().strip()
        
        intent_patterns = {
            'call': ['gọi', 'call', 'phone', 'điện thoại', 'facetime', 'video call'],
            'control-device': ['bật', 'tắt', 'điều chỉnh', 'turn', 'on', 'off', 'đèn', 'quạt', 'điều hòa'],
            'search-internet': ['tìm', 'search', 'kiếm', 'google', 'internet'],
            'search-youtube': ['youtube', 'yt', 'trên youtube', 'tìm youtube', 'tìm trên youtube', 'xem trên youtube'],
            'set-alarm': ['báo thức', 'alarm', 'nhắc', 'đặt', 'set'],
            'send-mess': ['gửi', 'nhắn', 'tin', 'message', 'sms', 'messenger'],
            'open-cam': ['camera', 'chụp', 'quay', 'ảnh', 'video'],
            'make-video-call': ['video call', 'facetime', 'gọi video'],
            'add-contacts': ['thêm', 'lưu', 'add', 'contact', 'danh bạ', 'số điện thoại'],
            'get-info': ['hỏi', 'kiểm tra', 'thông tin', 'info', 'time', 'pin']
        }
        
        intent_scores: Dict[str, int] = {}
        for intent, patterns in intent_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in text_lower:
                    score += 1
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda kv: kv[1])[0]
            confidence = min(0.9, 0.5 + (intent_scores[best_intent] * 0.1))
        else:
            best_intent = "unknown"
            confidence = 0.0
        
        entities = self._extract_entities_by_intent(text, best_intent)
        
        return {
            "intent": best_intent,
            "confidence": confidence,
            "entities": entities,
            "command": best_intent
        }
    
    def _extract_entities_by_intent(self, text: str, intent: str) -> Dict[str, str]:
        """Extract entities based on intent patterns"""
        entities = {}
        text_lower = text.lower()
        
        if intent == 'call':
            if 'mẹ' in text_lower:
                entities['CONTACT_NAME'] = 'mẹ'
            elif 'bố' in text_lower:
                entities['CONTACT_NAME'] = 'bố'
            elif 'bạn' in text_lower:
                entities['CONTACT_NAME'] = 'bạn'
            
            import re
            phone_pattern = r'\b\d{10,11}\b'
            phone_match = re.search(phone_pattern, text)
            if phone_match:
                entities['PHONE'] = phone_match.group()
        
        elif intent == 'control-device':
            if 'đèn' in text_lower:
                entities['DEVICE'] = 'đèn'
            elif 'quạt' in text_lower:
                entities['DEVICE'] = 'quạt'
            elif 'điều hòa' in text_lower:
                entities['DEVICE'] = 'điều hòa'
            
            if 'phòng khách' in text_lower:
                entities['LOCATION'] = 'phòng khách'
            elif 'phòng ngủ' in text_lower:
                entities['LOCATION'] = 'phòng ngủ'
        
        elif intent == 'set-alarm':
            import re
            time_pattern = r'(\d{1,2})[h:]\s*(\d{0,2})'
            time_match = re.search(time_pattern, text)
            if time_match:
                hour = time_match.group(1)
                minute = time_match.group(2) if time_match.group(2) else '00'
                entities['TIME'] = f"{hour}:{minute}"
        
        elif intent == 'send-mess':
            if 'mẹ' in text_lower:
                entities['RECEIVER'] = 'mẹ'
            elif 'bố' in text_lower:
                entities['RECEIVER'] = 'bố'
            elif 'bạn' in text_lower:
                entities['RECEIVER'] = 'bạn'

        elif intent == 'open-cam':
            if any(k in text_lower for k in ['chụp', 'ảnh', 'hình']):
                entities['CAMERA_TYPE'] = 'image'
            elif 'quay' in text_lower or 'video' in text_lower:
                entities['CAMERA_TYPE'] = 'video'
            if any(k in text_lower for k in ['mở', 'bật']):
                entities['ACTION'] = 'mở'
            elif any(k in text_lower for k in ['tắt', 'đóng']):
                entities['ACTION'] = 'tắt'
            if 'trước' in text_lower:
                entities['MODE'] = 'trước'
            elif 'sau' in text_lower:
                entities['MODE'] = 'sau'
            entities.pop('PLATFORM', None)

        elif intent == 'search-youtube':
            entities['PLATFORM'] = 'youtube'
            try:
                import re
                m = re.search(r'(?:tìm|search)\s+(?:trên\s+youtube\s+)?(.+)', text_lower)
                if m:
                    q = m.group(1).strip()
                    q = re.sub(r'\b(trên|youtube)\b', '', q).strip()
                    if q:
                        entities['QUERY'] = q
            except Exception:
                pass
        
        return entities
    
    def _reasoning_validate(self, text: str, context: Optional[Dict[str, Any]], model_result: Dict[str, Any]) -> Dict[str, Any]:
        """Secondary validation using reasoning engine"""
        if not self.reasoning_loaded or self.reasoning_engine is None:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "entities": {},
                "command": "unknown",
                "method": "reasoning_error",
                "error": "Reasoning engine not available"
            }
        
        try:
            self.stats['reasoning_predictions'] += 1
            
            # Use reasoning engine for validation
            assert self.reasoning_engine is not None
            reasoning_result = self.reasoning_engine.reasoning_predict(text, context)

            intent = reasoning_result.get('intent', 'unknown') or 'unknown'
            if intent == 'help':
                intent = 'unknown'

            command = reasoning_result.get('command', reasoning_result.get('intent', 'unknown')) or 'unknown'
            if command == 'help':
                command = 'unknown'

            entities = reasoning_result.get('entities', {}) or {}

            return {
                "intent": intent,
                "confidence": reasoning_result.get('confidence', 0.0),
                "entities": entities,
                "command": command,
                "method": "reasoning_engine",
                "reasoning_details": reasoning_result
            }
            
        except Exception as e:
            self.logger.error(f"Reasoning validation error: {e}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "entities": {},
                "command": "unknown",
                "method": "reasoning_error",
                "error": str(e)
            }
    
    def _hybrid_decision(self, model_result: Dict[str, Any], reasoning_result: Dict[str, Any], text: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Hybrid decision making - Model-first approach"""
        try:
            self.stats['hybrid_predictions'] += 1
            
            model_intent = model_result.get('intent', 'unknown')
            model_confidence = model_result.get('confidence', 0.0)
            model_entities = model_result.get('entities', {})
            
            reasoning_intent = reasoning_result.get('intent', 'unknown')
            reasoning_confidence = reasoning_result.get('confidence', 0.0)
            reasoning_entities = reasoning_result.get('entities', {})
            
            specialized_entities = {}
            if hasattr(self, 'specialized_extractor_loaded') and self.specialized_extractor_loaded:
                try:
                    specialized_entities = self.specialized_entity_extractor.extract_all_entities(text, model_intent)
                except Exception as e:
                    self.logger.warning(f"Specialized entity extraction failed: {e}")
            
            if model_confidence >= 0.7:
                final_intent = model_intent
                final_confidence = model_confidence
                final_entities = model_entities.copy()
                decision_reason = "high_model_confidence"
                
                if specialized_entities:
                    final_entities.update(specialized_entities)
                    decision_reason = "high_model_confidence_with_specialized_entities"
                
                if reasoning_intent != "unknown" and reasoning_confidence > 0.5:
                    final_entities.update(reasoning_entities)
                    decision_reason = "high_model_confidence_with_reasoning_enhancement"
                
            elif model_confidence >= 0.4:
                if reasoning_intent != "unknown" and reasoning_confidence > 0.6:
                    final_intent = reasoning_intent
                    final_confidence = reasoning_confidence
                    final_entities = reasoning_entities.copy()
                    decision_reason = "reasoning_validation_override"
                else:
                    final_intent = model_intent
                    final_confidence = model_confidence
                    final_entities = model_entities.copy()
                    decision_reason = "medium_model_confidence"
                
            else:
                if reasoning_intent != "unknown" and reasoning_confidence > 0.3:
                    final_intent = reasoning_intent
                    final_confidence = reasoning_confidence
                    final_entities = reasoning_entities.copy()
                    decision_reason = "reasoning_fallback"
                else:
                    final_intent = model_intent
                    final_confidence = model_confidence
                    final_entities = model_entities.copy()
                    decision_reason = "model_fallback"
            
            if specialized_entities:
                final_entities.update(specialized_entities)
            try:
                text_l = (text or "").lower()
                video_keywords = ["gọi video", "goi video", "video call", "facetime", "videochat", "video chat"]
                if any(k in text_l for k in video_keywords):
                    final_intent = "make-video-call"
                    decision_reason = "video_call_override"
            except Exception:
                pass

            try:
                text_l = (text or "").lower()
                text_norm = self._normalize_text(text_l)
                has_time_entity = "TIME" in final_entities or (specialized_entities and "TIME" in specialized_entities)
                time_regex = r"\b(\d{1,2})(?:[:h]\s*(\d{1,2}))?\b"
                has_time_pattern = bool(re.search(time_regex, text_l))
                alarm_keywords_vn = ["báo thức", "hẹn giờ", "đặt báo", "hẹn báo"]
                alarm_keywords_ascii = ["bao thuc", "hen gio", "dat bao", "hen bao"]
                period_keywords_vn = ["sáng", "trưa", "chiều", "tối", "đêm", "mai", "mốt", "ngày mai", "thứ", "cn", "chủ nhật"]
                period_keywords_ascii = ["sang", "trua", "chieu", "toi", "dem", "mai", "mot", "ngay mai", "thu", "chu nhat", "cn"]
                has_alarm_kw = any(w in text_l for w in alarm_keywords_vn) or any(w in text_norm for w in alarm_keywords_ascii)
                has_period_kw = any(w in text_l for w in period_keywords_vn) or any(w in text_norm for w in period_keywords_ascii)
                comm_keywords = [
                    "gọi", "goi", "video", "call",
                    "nhắn", "nhan", "gửi", "gui",
                    "liên lạc", "lien lac", "nhắn tin", "nhan tin", "tin nhắn", "sms"
                ]
                receiver_keywords = [
                    "cho bà", "cho me", "cho mẹ", "cho ba", "cho bố", "cho ong", "cho ông",
                    "cho anh", "cho chi", "cho chị", "cho em", "cho con",
                    "tới", "đến", "toi", "den"
                ]
                is_communication = any(w in text_l for w in comm_keywords) or any(k in text_l for k in receiver_keywords)
                if is_communication and final_intent not in ["send-mess", "call", "make-video-call"]:
                    final_intent = "send-mess"
                    decision_reason = "communication_guard_send_mess"
                elif final_intent != "set-alarm" and not is_communication and (has_time_entity or (has_time_pattern and (has_alarm_kw or has_period_kw))):
                    final_intent = "set-alarm"
                    decision_reason = "time_override_set_alarm"
            except Exception:
                pass

            try:
                text_l = (text or "").lower()
                text_norm = self._normalize_text(text_l)
                contact_keywords_vn = ["liên hệ", "liên lạc", "danh bạ", "thêm số", "lưu số", "thêm liên hệ", "tạo liên hệ"]
                contact_keywords_ascii = ["lien he", "lien lac", "danh ba", "them so", "luu so", "them lien he", "tao lien he", "add contact"]
                has_contact_kw = any(k in text_l for k in contact_keywords_vn) or any(k in text_norm for k in contact_keywords_ascii)
                has_digits_phone = bool(re.search(r"\b\d{9,11}\b", text_l))
                has_word_phone = bool(re.search(r"\b(không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín)\b", text_l)) or bool(re.search(r"\b(khong|mot|hai|ba|bon|nam|sau|bay|tam|chin)\b", text_norm))
                if has_contact_kw and (has_digits_phone or has_word_phone):
                    final_intent = "add-contacts"
            except Exception:
                pass

            try:
                text_l = (text or "").lower()
                if any(w in text_l for w in ["youtube", "yt"]):
                    final_intent = "search-youtube"
                    if hasattr(self, 'specialized_extractor_loaded') and self.specialized_extractor_loaded:
                        try:
                            se = self.specialized_entity_extractor._extract_youtube_search_entities(text)  # noqa: SLF001
                            if se and isinstance(se, dict):
                                final_entities.update(se)
                        except Exception:
                            pass
            except Exception:
                pass

            try:
                text_l = (text or "").lower()
                text_norm = self._normalize_text(text_l)
                search_keywords_vn = ["tra cứu", "tìm", "bảng giá", "giá", "thông tin", "đánh giá", "tìm kiếm", "tìm hiểu", "tra", "tra cứu", "sớt"]
                search_keywords_ascii = ["tra cuu", "tim", "bang gia", "gia", "thong tin", "danh gia", "search", "spec"]
                device_keywords_vn = ["độ sáng", "ánh sáng", "âm lượng", "âm thanh", "loa", "đèn", "điều hòa"]
                device_keywords_ascii = ["do sang", "anh sang", "am luong", "am thanh", "loa", "den", "dieu hoa", "wifi", "wi fi", "wi-fi", "brightness", "volume"]
                has_search_kw = any(k in text_l for k in search_keywords_vn) or any(k in text_norm for k in search_keywords_ascii)
                has_device_kw = any(k in text_l for k in device_keywords_vn) or any(k in text_norm for k in device_keywords_ascii)
                if has_search_kw and not has_device_kw and final_intent not in ["search-youtube", "add-contacts", "call", "make-video-call", "send-mess"] and not any(tok in text_l for tok in ["youtube", "yt"]):
                    final_intent = "search-internet"
                    try:
                        if hasattr(self, 'specialized_extractor_loaded') and self.specialized_extractor_loaded:
                            se = self.specialized_entity_extractor.extract_search_entities(text)  # type: ignore[attr-defined]
                            if se and isinstance(se, dict):
                                final_entities.update(se)
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                text_l = (text or "").lower()
                if any(w in text_l for w in ["đèn pin", "den pin", "flash", "flashlight"]):
                    final_intent = "control-device"
                    final_entities["DEVICE"] = "flash"
                    if any(w in text_l for w in ["bật", "mở", "on"]):
                        final_entities["ACTION"] = "ON"
                    elif any(w in text_l for w in ["tắt", "đóng", "off"]):
                        final_entities["ACTION"] = "OFF"
                elif any(w in text_l for w in ["wifi", "wi fi", "wi-fi", "wiai phai", "wai phai"]):
                    final_intent = "control-device"
                    final_entities["DEVICE"] = "wifi"
                    if any(w in text_l for w in ["bật", "mở", "on"]):
                        final_entities["ACTION"] = "ON"
                    elif any(w in text_l for w in ["tắt", "đóng", "off"]):
                        final_entities["ACTION"] = "OFF"
                elif any(w in text_l for w in ["bluetooth", "blue tooth", "blutooth"]):
                    final_intent = "control-device"
                    final_entities["DEVICE"] = "bluetooth"
                    if any(w in text_l for w in ["bật", "mở", "on"]):
                        final_entities["ACTION"] = "ON"
                    elif any(w in text_l for w in ["tắt", "đóng", "off"]):
                        final_entities["ACTION"] = "OFF"
                elif any(w in text_l for w in ["data", "3g", "4g", "5g", "dữ liệu", "du lieu"]):
                    final_intent = "control-device"
                    final_entities["DEVICE"] = "mobile_data"
                    if any(w in text_l for w in ["bật", "mở", "on"]):
                        final_entities["ACTION"] = "ON"
                    elif any(w in text_l for w in ["tắt", "đóng", "off"]):
                        final_entities["ACTION"] = "OFF"
                elif any(w in text_l for w in ["độ sáng", "brightness"]):
                    final_intent = "control-device"
                    final_entities["DEVICE"] = "brightness"
                    if any(w in text_l for w in ["bật", "mở", "on"]):
                        final_entities["ACTION"] = "ON"
                    elif any(w in text_l for w in ["tắt", "đóng", "off"]):
                        final_entities["ACTION"] = "OFF"
                elif any(w in text_l for w in ["âm lượng", "volume", "âm thanh", "tiếng"]):
                    final_intent = "control-device"
                    final_entities["DEVICE"] = "volume"
                    if any(w in text_l for w in ["bật", "mở", "on"]):
                        final_entities["ACTION"] = "ON"
                    elif any(w in text_l for w in ["tắt", "đóng", "off"]):
                        final_entities["ACTION"] = "OFF"
            except Exception:
                pass

            try:
                if hasattr(self, 'specialized_extractor_loaded') and self.specialized_extractor_loaded:
                    refined_entities = self.specialized_entity_extractor.extract_all_entities(text, final_intent)
                    if refined_entities:
                        final_entities.update(refined_entities)
                    if final_intent == "set-alarm":
                        try:
                            alarm_entities = self.specialized_entity_extractor._extract_alarm_entities(text)  # noqa: SLF001
                            if alarm_entities:
                                final_entities.update(alarm_entities)
                        except Exception:
                            pass
            except Exception as e:
                self.logger.warning(f"Refined entity extraction failed: {e}")

            try:
                if final_intent in ["call", "make-video-call"]:
                    if "RECEIVER" not in final_entities and "RECEIVER" in reasoning_entities:
                        final_entities["RECEIVER"] = reasoning_entities["RECEIVER"]
                    if "CONTACT_NAME" not in final_entities and "CONTACT_NAME" in reasoning_entities:
                        final_entities["CONTACT_NAME"] = reasoning_entities["CONTACT_NAME"]
            except Exception:
                pass

            nlp_response = None
            unknown_threshold = 0.35
            if final_confidence < unknown_threshold or final_intent == "unknown":
                final_intent = "unknown"
                final_entities = {}
                final_confidence = max(final_confidence, 0.0)
                decision_reason = f"fallback_unknown_threshold_{unknown_threshold}"
                nlp_response = "Hệ thống chưa đủ chắc chắn, vui lòng nói rõ hơn hoặc nhắc lại giúp."
            valid_commands = getattr(self, "valid_commands", None)
            if valid_commands:
                final_command = final_intent if final_intent in valid_commands else "unknown"
            else:
                final_command = final_intent
            
            return {
                "intent": final_intent,
                "confidence": final_confidence,
                "entities": final_entities,
                "command": final_command,
                "method": "hybrid",
                "decision_reason": decision_reason,
                "primary_source": "trained_model" if model_confidence >= 0.4 else "reasoning_engine",
                "nlp_response": nlp_response,
            }
            
        except Exception as e:
            self.logger.error(f"Hybrid decision error: {e}")
            return model_result
    
    def _fallback_prediction(self, text: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback prediction when all else fails"""
        self.stats['fallback_predictions'] += 1
        
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "entities": {},
            "command": "unknown",
            "method": "fallback",
            "input_text": text,
            "context": context or {},
            "timestamp": time.time(),
            "error": "All prediction methods failed"
        }
    
    def _update_stats(self, result: Dict[str, Any], processing_time: float):
        """Update performance statistics"""
        try:
            total_predictions = self.stats['total_predictions']
            current_avg = self.stats['avg_processing_time']
            self.stats['avg_processing_time'] = (current_avg * (total_predictions - 1) + processing_time) / total_predictions
            
            confidence = result.get('confidence', 0.0)
            self.stats['confidence_scores'].append(confidence)
            
            if len(self.stats['confidence_scores']) > 100:
                self.stats['confidence_scores'] = self.stats['confidence_scores'][-100:]
                
        except Exception as e:
            self.logger.error(f"Stats update error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        stats = self.stats.copy()
        
        if stats['confidence_scores']:
            stats['avg_confidence'] = sum(stats['confidence_scores']) / len(stats['confidence_scores'])
            stats['min_confidence'] = min(stats['confidence_scores'])
            stats['max_confidence'] = max(stats['confidence_scores'])
        
        stats['success_rate'] = (stats['total_predictions'] - stats['fallback_predictions']) / max(stats['total_predictions'], 1)
        
        return stats
    
    def test_system(self, test_cases: List[str]) -> Dict[str, Any]:
        """Test the hybrid system with multiple test cases"""
        results = []
        
        for test_case in test_cases:
            result = self.predict(test_case)
            results.append({
                "input": test_case,
                "result": result
            })
        
        analysis = {
            "total_tests": len(test_cases),
            "successful_predictions": len([r for r in results if r["result"]["intent"] != "unknown"]),
            "avg_confidence": sum([r["result"]["confidence"] for r in results]) / len(results),
            "avg_processing_time": sum([r["result"].get("processing_time", 0) for r in results]) / len(results),
            "results": results
        }
        
        return analysis
