import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import json
import os
import yaml
import logging
import faiss
from collections import defaultdict, deque
from rapidfuzz import fuzz, process
import time
from src.training.configs.config import ModelConfig, IntentConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reasoning_engine.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ReasoningEngine")

class ReasoningCache:
    """Cache cho các embedding và kết quả tính toán"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.embedding_cache = {}
        self.similarity_cache = {}
        self.result_cache = {}
        self.access_history = deque(maxlen=max_size)
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Lấy embedding từ cache"""
        return self.embedding_cache.get(text)
    
    def set_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Lưu embedding vào cache"""
        self._manage_cache_size(self.embedding_cache)
        self.embedding_cache[text] = embedding
        self._update_access_history(text)
    
    def get_similarity(self, text_pair: Tuple[str, str]) -> Optional[float]:
        """Lấy similarity từ cache"""
        return self.similarity_cache.get(text_pair)
    
    def set_similarity(self, text_pair: Tuple[str, str], similarity: float) -> None:
        """Lưu similarity vào cache"""
        self._manage_cache_size(self.similarity_cache)
        self.similarity_cache[text_pair] = similarity
    
    def get_result(self, text: str) -> Optional[Dict]:
        """Lấy kết quả từ cache"""
        return self.result_cache.get(text)
    
    def set_result(self, text: str, result: Dict) -> None:
        """Lưu kết quả vào cache"""
        self._manage_cache_size(self.result_cache)
        self.result_cache[text] = result
        self._update_access_history(text)
    
    def _update_access_history(self, key: str) -> None:
        """Cập nhật lịch sử truy cập"""
        if key in self.access_history:
            self.access_history.remove(key)
        self.access_history.append(key)
    
    def _manage_cache_size(self, cache_dict: Dict) -> None:
        """Quản lý kích thước cache"""
        if len(cache_dict) >= self.max_size:
            oldest = self.access_history.popleft()
            if oldest in cache_dict:
                del cache_dict[oldest]
    
    def clear(self) -> None:
        """Xóa toàn bộ cache"""
        self.embedding_cache.clear()
        self.similarity_cache.clear()
        self.result_cache.clear()
        self.access_history.clear()

class FuzzyMatcher:
    """Fuzzy matching cho các từ khóa và pattern"""
    
    def __init__(self, threshold: int = 75):
        self.threshold = threshold
    
    def match(self, query: str, choices: List[str]) -> List[Tuple[str, int]]:
        """Tìm các match với score cao nhất"""
        results = process.extract(
            query, 
            choices, 
            scorer=fuzz.token_sort_ratio, 
            score_cutoff=self.threshold
        )
        # process.extract returns List[Tuple[str, float, int]], convert to List[Tuple[str, int]]
        return [(text, idx) for text, score, idx in results]
    
    def contains_fuzzy(self, text: str, keywords: List[str], threshold: Optional[int] = None) -> List[Tuple[str, int]]:
        """Kiểm tra xem text có chứa bất kỳ keyword nào không bằng rapidfuzz (fuzzy matching)"""
        match_threshold = threshold if threshold is not None else self.threshold
        matches = []
        
        for keyword in keywords:
            word_length = len(keyword)
            min_length = max(3, word_length - 2)
            max_length = word_length + 2
            
            words = text.lower().split()
            for word in words:
                if min_length <= len(word) <= max_length:
                    score = fuzz.ratio(keyword, word)
                    if score >= match_threshold:
                        matches.append((keyword, score))
        
        return matches

class VectorStore:
    """Vector store sử dụng FAISS cho tìm kiếm semantic"""
    
    def __init__(self, vector_dim: int = 1024):
        self.vector_dim = vector_dim
        self.index = faiss.IndexFlatIP(vector_dim)  # Inner product để cosine similarity
        self.text_mapping = []  # Lưu text tương ứng với các vector
        
    def add_vectors(self, texts: List[str], vectors: np.ndarray) -> None:
        """Thêm vectors vào index"""
        vectors_normalized = vectors.astype(np.float32)
        # Normalize L2: faiss.normalize_L2 modifies in-place
        # FAISS normalize_L2 modifies array in-place, type checker may not recognize the signature
        faiss.normalize_L2(vectors_normalized)  # pyright: ignore[reportCallIssue]
        self.index.add(vectors_normalized)
        self.text_mapping.extend(texts)
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Tìm kiếm vectors gần nhất"""
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_vector)  # type: ignore
        
        distances, indices = self.index.search(query_vector, top_k)  # type: ignore
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.text_mapping):  # Check valid index
                results.append((self.text_mapping[idx], float(distances[0][i])))
        
        return results
    
    def reset(self) -> None:
        """Reset index"""
        self.index = faiss.IndexFlatIP(self.vector_dim)
        self.text_mapping = []

class ConversationContext:
    """Quản lý ngữ cảnh hội thoại Nhiệm vụ: hỗ trợ multi-turn (ví dụ turn trước đang set-alarm, turn sau chỉ nói “7 giờ”)."""
    
    def __init__(self, max_history: int = 5):
        self.history = deque(maxlen=max_history)
        self.current_intent = None
        self.current_entities = {}
        self.session_data = {}
    
    def add_turn(self, user_input: str, system_response: Dict[str, Any]) -> None:
        """Thêm một lượt đối thoại vào history"""
        self.history.append({
            "user_input": user_input,
            "system_response": system_response,
            "timestamp": time.time()
        })
        
        # Sửa lỗi: system_response có thể là string hoặc dict
        if isinstance(system_response, dict):
            if system_response.get("intent") and system_response.get("confidence", 0) > 0.5:
                self.current_intent = system_response["intent"]
            
            if system_response.get("entities"):
                self.current_entities.update(system_response["entities"])
        else:
            logger.warning(f"system_response is not a dict: {type(system_response)} - {system_response}")
    
    def get_last_n_turns(self, n: int = 3) -> List[Dict[str, Any]]:
        """Lấy n lượt đối thoại gần nhất"""
        return list(self.history)[-n:] if self.history else []
    
    def get_current_context(self) -> Dict[str, Any]:
        """Lấy ngữ cảnh hiện tại"""
        return {
            "current_intent": self.current_intent,
            "current_entities": self.current_entities,
            "session_data": self.session_data
        }
    
    def reset(self) -> None:
        """Reset ngữ cảnh"""
        self.history.clear()
        self.current_intent = None
        self.current_entities = {}
        self.session_data = {}

class EntityExtractor:
    """Trích xuất entities từ văn bản"""
    
    def __init__(self, fuzzy_matcher: FuzzyMatcher):
        self.fuzzy_matcher = fuzzy_matcher
        self.entity_patterns = self._load_entity_patterns()
    
    def _load_entity_patterns(self) -> Dict[str, Dict]:
        """Load entity patterns"""
        return {
            "time": {
                "patterns": [
                    r"(\d{1,2})[h:]\s*(\d{1,2})?\s*(sáng|chiều|tối|đêm)?",  # 8h30, 8:30, 6h chiều
                    r"(\d{1,2})\s*(giờ|tiếng)\s*(\d{1,2})?\s*(phút)?\s*(sáng|chiều|tối|đêm)?",  # 8 giờ 30 phút sáng
                    r"(sáng|trưa|chiều|tối|đêm)",  # sáng, chiều, tối
                    r"(hôm nay|ngày mai|hôm qua)",  # hôm nay, ngày mai
                ],
                "keywords": ["giờ", "phút", "sáng", "trưa", "chiều", "tối", "đêm", 
                             "hôm nay", "ngày mai", "hôm qua", "tuần", "tháng"]
            },
                "person": {
                "patterns": [
                    r"(?:cho|tới|đến|với)\s+((?:ba|bố|mẹ|anh|chị|em|cô|chú|bác|ông|bà)\s+[\w\s]+?)(?:\s+rằng|\s+là|\s+nói|\s+nhắn|\s+gửi|\s+lúc|\s+tại|\s+ở|\s+vào|\s+ngày|\s+giờ|$)",
                    r"(mẹ|ba|bố|bạn|anh|chị|em|cô|chú|bác|ông|bà)\s*(của|tôi|tui|mình)?",
                ],
                "keywords": ["mẹ", "ba", "bố", "bạn", "anh", "chị", "em", "cô", "chú", "bác", "ông", "bà"]
            },
                "location": {
                "patterns": [
                    r"(?:tại|ở)\s+([^lúc]*?)(?:\s+lúc|\s+giờ|\s+vào|\s+ngày|$)",
                    r"(bệnh viện|trường|công viên|nhà|công ty|văn phòng|phòng)\s+([\w\s]+?)(?:\s+lúc|\s+giờ|\s+vào|\s+ngày|,|$)",
                    r"(nhà|công ty|văn phòng|bệnh viện|trường|phòng|công viên)",
                ],
                "keywords": ["nhà", "công ty", "văn phòng", "bệnh viện", "trường", "phòng", "quán", "công viên", "tại", "ở"]
            }
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Trích xuất entities từ text"""
        entities = {entity_type: [] for entity_type in self.entity_patterns}
        
        for entity_type, patterns_data in self.entity_patterns.items():
            for pattern in patterns_data["patterns"]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple):
                            entity_value = " ".join([m for m in match if m])
                            if entity_value:
                                entity_value = entity_value.strip()
                                entity_value = self._clean_entity_value(entity_type, entity_value)
                                if entity_value:
                                    entities[entity_type].append(entity_value)
                        else:
                            entity_value = match.strip()
                            entity_value = self._clean_entity_value(entity_type, entity_value)
                            if entity_value:
                                entities[entity_type].append(entity_value)
            
            if not entities[entity_type]:
                fuzzy_matches = self.fuzzy_matcher.contains_fuzzy(text, patterns_data["keywords"])
                for keyword, _ in fuzzy_matches:
                    keyword_idx = text.lower().find(keyword.lower())
                    if keyword_idx >= 0:
                        start_idx = max(0, text.rfind(" ", 0, keyword_idx) + 1)
                        end_idx = text.find(" ", keyword_idx + len(keyword))
                        if end_idx == -1:
                            end_idx = len(text)
                        
                        phrase = text[start_idx:end_idx].strip()
                        phrase = self._clean_entity_value(entity_type, phrase)
                        if phrase and phrase not in entities[entity_type]:
                            entities[entity_type].append(phrase)
        
        for entity_type in entities:
            entities[entity_type] = list(set(entities[entity_type]))
            entities[entity_type].sort(key=len)
        
        return entities
    
    def _clean_entity_value(self, entity_type: str, value: str) -> str:
        """Làm sạch entity value"""
        if not value:
            return ""
        
        stop_words = ["rằng", "là", "nói", "nhắn", "gửi", "cho", "tới", "đến", "với"]
        words = value.split()
        cleaned_words = [word for word in words if word.lower() not in stop_words]
        
        if entity_type == "person":
            if len(cleaned_words) > 1:
                return " ".join(cleaned_words)
            else:
                person_words = []
                for word in cleaned_words:
                    if word.lower() in ["mẹ", "ba", "bố", "bạn", "anh", "chị", "em", "cô", "chú", "bác", "ông", "bà", "tôi", "tui", "mình"]:
                        person_words.append(word)
                return " ".join(person_words)
        
        elif entity_type == "time":
            time_parts = []
            for word in cleaned_words:
                if word.lower() in ["giờ", "phút", "sáng", "trưa", "chiều", "tối", "đêm", "hôm nay", "ngày mai", "hôm qua", "tuần", "tháng"] or word.isdigit():
                    time_parts.append(word)
            return " ".join(time_parts)
        
        elif entity_type == "location":
            time_words = ["lúc", "giờ", "vào", "ngày", "sáng", "chiều", "tối", "đêm"]
            location_words = [word for word in cleaned_words if word.lower() not in time_words]
            
            location_types = ["công viên", "bệnh viện", "trường", "nhà", "công ty", "văn phòng", "phòng"]
            for loc_type in location_types:
                if loc_type in value.lower():
                    words = value.split()
                    loc_index = -1
                    for i, word in enumerate(words):
                        if loc_type in word.lower():
                            loc_index = i
                            break
                    
                    if loc_index >= 0:
                        location_parts = [words[loc_index]]
                        for i in range(loc_index + 1, len(words)):
                            if words[i].lower() in time_words or words[i] in [",", ".", "!"]:
                                break
                            location_parts.append(words[i])
                        return " ".join(location_parts)
            
            return " ".join(location_words)
        
        return " ".join(cleaned_words)
    
    def _convert_words_to_numbers(self, text: str) -> str:
        """Convert số từ chữ sang số"""
        result = text.lower()
        
        number_words = {
            'không': '0', 'một': '1', 'hai': '2', 'ba': '3', 'bốn': '4',
            'năm': '5', 'sáu': '6', 'bảy': '7', 'tám': '8', 'chín': '9',
            'mốt': '1', 'oan': '1', 'one': '1', 'mot': '1',
            'tu': '2', 'two': '2',
            'tư': '4', 'four': '4',
            'lăm': '5', 'five': '5', 'lam': '5',
            'sáu': '6', 'six': '6', 'sau': '6',
            'bảy': '7', 'seven': '7', 'bay': '7',
            'tám': '8', 'eight': '8', 'tam': '8',
            'chín': '9', 'nine': '9', 'chin': '9',
            'zê rô': '0', 'zero': '0', 'ze ro': '0', 'khong': '0',
        }
        
        for word, number in sorted(number_words.items(), key=lambda x: len(x[0]), reverse=True):
            result = result.replace(word, number)
        
        return result

class ReasoningEngine:
    """Hệ thống tự suy luận sử dụng PhoBERT"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.cache = ReasoningCache(max_size=2000)
        
        # Đọc config từ file nếu có
        self.config = self._load_config(config_path)
        
        # Initialize variables to avoid "possibly unbound" errors
        model_name = self.config.get("model_name", "vinai/phobert-large")
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        try:
            # Load PhoBERT with correct settings - SAME AS TRAINING SYSTEM
            # Try loading with local cache first (same as training system)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    use_fast=False,
                    cache_dir=cache_dir,
                    local_files_only=True
                )
                self.model = AutoModel.from_pretrained(
                    model_name,
                    use_safetensors=True,
                    trust_remote_code=False,
                    cache_dir=cache_dir,
                    local_files_only=True
                )
                self.model.eval()
            except Exception:
                snapshots_dir = os.path.join(cache_dir, "models--vinai--phobert-large", "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshot_dirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                    model_path = None
                    for snapshot_dir in snapshot_dirs:
                        snapshot_path = os.path.join(snapshots_dir, snapshot_dir)
                        if os.path.exists(os.path.join(snapshot_path, "model.safetensors")):
                            model_path = snapshot_path
                            break
                    
                    if model_path:
                        try:
                            self.tokenizer = AutoTokenizer.from_pretrained(
                                model_name, 
                                use_fast=False,
                                cache_dir=cache_dir,
                                local_files_only=False  # Allow download for tokenizer
                            )
                        except:
                            self.tokenizer = AutoTokenizer.from_pretrained(
                                model_path, 
                                use_fast=False,
                                local_files_only=True
                            )
                        
                        self.model = AutoModel.from_pretrained(
                            model_path,
                            use_safetensors=True,
                            trust_remote_code=False,
                            local_files_only=True
                        )
                        self.model.eval()
                    else:
                        raise Exception("No snapshot with model.safetensors found in cache")
                else:
                    raise Exception("Model not found in cache")
        except Exception as e:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    use_fast=False,
                    cache_dir=cache_dir,
                    local_files_only=False
                )
                self.model = AutoModel.from_pretrained(
                    model_name,
                    use_safetensors=True,
                    trust_remote_code=False,
                    cache_dir=cache_dir,
                    local_files_only=False
                )
                self.model.eval()
            except Exception as e2:
                logger.error(f"Error loading model: {e2}")
                self.tokenizer = None
                self.model = None
                logger.warning("Using fallback embedding mode - reasoning engine will work with reduced accuracy")
        
        self.fuzzy_matcher = FuzzyMatcher(threshold=self.config.get("fuzzy_threshold", 75))
        
        try:
            if self.model is not None:
                self.vector_store = VectorStore(vector_dim=self.model.config.hidden_size)
            else:
                self.vector_store = VectorStore(vector_dim=768)
        except Exception as e:
            logger.warning(f"Vector store initialization failed: {e}")
            self.vector_store = None
        
        try:
            from src.inference.engines.entity_extractor import EntityExtractor as SpecializedEntityExtractor
            self.entity_extractor = SpecializedEntityExtractor()
        except ImportError:
            self.entity_extractor = EntityExtractor(self.fuzzy_matcher)
            logger.warning("Using basic EntityExtractor as fallback")
        
        self.conversation_context = ConversationContext(max_history=self.config.get("max_history", 5))
        
        self.knowledge_base = self._load_knowledge_base(
            os.path.join(os.path.dirname(__file__), "knowledge_base.json")
        )
        
        self.semantic_patterns = self._load_semantic_patterns(
            os.path.join(os.path.dirname(__file__), "semantic_patterns.json")
        )
        
        self.context_rules = self._load_context_rules(
            os.path.join(os.path.dirname(__file__), "context_rules.json")
        )
        
        self.intent_fallback = self._load_intent_fallback(
            self.config.get("fallback_path", os.path.join(os.path.dirname(__file__), "intent_fallback.json"))
        )
        
        self._initialize_vector_store()
        
        self.similarity_threshold = self.config.get("similarity_threshold", 0.6)
        
        self.normalize_intent = {
            "call": "call",
            "send-mess": "send-mess", 
            "make-video-call": "make-video-call",
            "search-internet": "search-internet",
            "search-youtube": "search-youtube",
            "get-info": "get-info",
            "set-alarm": "set-alarm",
            "open-cam": "open-cam",
            "control-device": "control-device",
            "add-contacts": "add-contacts",
            "unknown": "unknown",
            "check-weather": "get-info",
            "read-news": "get-info", 
            "check-health-status": "get-info",
            "general-conversation": "unknown",
            "help": "unknown"
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load config từ file"""
        default_config = {
            "model_name": "vinai/phobert-large",
            "max_length": 256,
            "similarity_threshold": 0.6,
            "fuzzy_threshold": 75,
            "max_history": 5,
            "knowledge_base_path": os.path.join(os.path.dirname(__file__), "knowledge_base.json"),
            "patterns_path": os.path.join(os.path.dirname(__file__), "semantic_patterns.json"),
            "rules_path": os.path.join(os.path.dirname(__file__), "context_rules.json"),
            "fallback_path": os.path.join(os.path.dirname(__file__), "intent_fallback.json"),
            "enable_fuzzy_matching": True,
            "enable_vectorstore": True,
            "enable_cache": True,
            "pooling_strategy": "mean"  # 'cls', 'mean', or 'max'
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.json'):
                    loaded_config = json.load(f)
                elif config_path.endswith(('.yaml', '.yml')):
                    loaded_config = yaml.safe_load(f)
                else:
                    loaded_config = {}
                    logger.warning(f"Không hỗ trợ định dạng file config: {config_path}")
                
                # Cập nhật config
                default_config.update(loaded_config)
        
        return default_config
    
    def _load_knowledge_base(self, file_path: str) -> Dict[str, Any]:
        """Load knowledge base từ file hoặc sử dụng mặc định"""
        default_kb = {
            "intent_synonyms": {
                # 13 command chuẩn
                "call": ["gọi", "điện thoại", "alo", "kết nối", "liên lạc", "quay số", "bấm số", "phone", "call"],
                "make-video-call": ["gọi video", "video call", "face time", "zalo video", "gọi facetime"],
                "send-mess": ["gửi tin nhắn", "nhắn tin", "text", "sms", "thông báo", "soạn tin", "message", "send"],
                "add-contacts": ["thêm liên lạc", "lưu số", "thêm số", "lưu danh bạ", "thêm danh bạ"],
                "search-internet": ["tìm kiếm", "tìm", "search", "tra cứu", "google"],
                "search-youtube": ["tìm youtube", "tìm trên youtube", "youtube", "yt"],
                "get-info": ["thông tin", "thời tiết", "nhiệt độ", "mưa", "nắng", "dự báo thời tiết", "weather", "temperature", 
                           "đọc tin tức", "tin tức", "báo", "thời sự", "cập nhật tin", "news", "read",
                           "kiểm tra sức khỏe", "đo", "theo dõi", "chỉ số", "tình trạng", "health", "check"],
                "set-alarm": ["đặt báo thức", "nhắc nhở", "hẹn giờ", "đánh thức", "chuông báo", "ghi nhớ", "alarm", "reminder"],
                "open-cam": ["mở camera", "bật camera", "camera", "chụp ảnh"],
                "control-device": ["điều khiển", "bật", "tắt", "wifi", "bluetooth", "đèn pin", "âm lượng", "volume"],
                "unknown": ["không hiểu", "không rõ", "lạ", "không biết", "chưa rõ", "xin chào", "tạm biệt", "cảm ơn", "trò chuyện", "nói chuyện", "hello", "conversation",
                          "giúp đỡ", "trợ giúp", "hướng dẫn", "hỗ trợ", "không hiểu", "help", "support"]
            },
            "context_keywords": {
                "time": ["giờ", "phút", "sáng", "chiều", "tối", "mai", "hôm nay", "tuần", "tháng"],
                "person": ["mẹ", "bố", "con", "cháu", "bạn", "anh", "chị", "em", "ông", "bà"],
                "location": ["nhà", "bệnh viện", "phòng", "ngoài", "trong", "đây", "đó"],
                "action": ["uống", "ăn", "ngủ", "đi", "về", "đến", "gặp", "thăm"],
                "object": ["thuốc", "nước", "cơm", "sách", "điện thoại", "tivi", "radio"]
            },
            "intent_indicators": {
                # 13 command chuẩn
                "call": ["gọi", "điện", "phone", "call", "kết nối", "liên lạc", "cuộc gọi", "gọi thoại", "gọi điện", "thực hiện gọi", "thực hiện cuộc gọi"],
                "make-video-call": ["gọi video", "video call", "face time", "zalo video", "gọi facetime", "video"],
                "send-mess": ["nhắn", "tin", "message", "sms", "text", "gửi", "nhắn tin", "gửi tin", "soạn tin", "tin nhắn"],
                "add-contacts": ["thêm liên lạc", "lưu số", "thêm số", "lưu danh bạ", "thêm danh bạ", "lưu", "thêm"],
                "search-internet": ["tìm kiếm", "tìm", "search", "tra cứu", "google", "internet"],
                "search-youtube": ["youtube", "yt", "tìm youtube", "tìm trên youtube"],
                "get-info": ["thông tin", "thời tiết", "weather", "nhiệt", "mưa", "nắng", "tin", "news", "báo", "đọc", "thời sự", 
                            "sức khỏe", "health", "kiểm tra", "đo", "theo dõi", "chỉ số", "tình trạng"],
                "set-alarm": ["báo thức", "nhắc", "hẹn", "alarm", "reminder", "giờ", "chuông"],
                "open-cam": ["camera", "chụp", "ảnh", "mở camera", "bật camera"],
                "control-device": ["điều khiển", "bật", "tắt", "wifi", "bluetooth", "đèn pin", "âm lượng", "volume", "thiết bị"],
                "unknown": ["không hiểu", "chưa rõ", "không biết", "chào", "hello", "cảm ơn", "tạm biệt", "nói chuyện", 
                          "giúp", "help", "hướng dẫn", "hỗ trợ"]
            },
            "multi_intent_indicators": {
                "call,send-mess": ["gọi", "nhắn", "liên lạc"],
                "get-info,search-internet": ["thông tin", "tìm kiếm", "cập nhật"]
            }
        }
        
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_kb = json.load(f)
                    for key, value in loaded_kb.items():
                        if key in default_kb:
                            if isinstance(value, dict):
                                default_kb[key].update(value)
                            else:
                                default_kb[key] = value
                        else:
                            default_kb[key] = value
                    
            except Exception as e:
                logger.error(f"Lỗi khi load knowledge base: {str(e)}")
        else:
            if file_path and os.path.dirname(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(default_kb, f, ensure_ascii=False, indent=2)
        
        return default_kb
    
    def _load_semantic_patterns(self, file_path: str) -> Dict[str, List[str]]:
        """Load semantic patterns từ file hoặc sử dụng mặc định"""
        default_patterns = {
            "call_patterns": [
                r"gọi.*(?:cho|tới|đến)",
                r"(?:điện thoại|phone).*(?:cho|tới)",
                r"kết nối.*(?:với|cho)",
                r"liên lạc.*(?:với|cho)",
                r"cuộc gọi.*(?:cho|tới|đến)",
                r"gọi thoại.*(?:cho|tới|đến)",
                r"thực hiện.*(?:cuộc gọi|gọi).*(?:cho|tới|đến)",
                r"gọi điện.*(?:cho|tới|đến)"
            ],
            "alarm_patterns": [
                r"đặt.*(?:báo thức|nhắc nhở)",
                r"hẹn.*(?:giờ|thời gian)",
                r"nhắc.*(?:tôi|lúc|khi)",
                r"báo thức.*(?:lúc|giờ)"
            ],
            "message_patterns": [
                r"gửi.*(?:tin nhắn|tin)",
                r"nhắn.*(?:tin|cho)",
                r"soạn.*(?:tin nhắn|tin)",
                r"text.*(?:cho|tới)",
                r"vào\s+\w+.*nhắn.*cho",
                r"nhắn\s+cho.*rằng",
                r"gửi\s+cho.*rằng"
            ],
            "weather_patterns": [
                r"thời tiết.*(?:hôm nay|mai|thế nào)",
                r"nhiệt độ.*(?:bao nhiêu|thế nào)",
                r"dự báo.*(?:thời tiết|mưa|nắng)"
            ],
            "news_patterns": [
                r"đọc.*(?:tin tức|báo)",
                r"tin tức.*(?:mới nhất|hôm nay)",
                r"báo.*(?:hôm nay|mới nhất)"
            ],
            "health_patterns": [
                r"kiểm tra.*(?:sức khỏe|huyết áp)",
                r"đo.*(?:huyết áp|nhịp tim)",
                r"theo dõi.*(?:sức khỏe|tình trạng)"
            ],
            "conversation_patterns": [
                r"xin chào",
                r"chào.*(?:bạn|anh|chị)",
                r"cảm ơn.*(?:bạn|anh|chị)",
                r"tạm biệt"
            ],
            "help_patterns": [
                r"giúp.*(?:tôi|đỡ|với)",
                r"hướng dẫn.*(?:tôi|cách)",
                r"hỗ trợ.*(?:tôi|với)"
            ]
        }
        
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_patterns = json.load(f)
                    for key, value in loaded_patterns.items():
                        default_patterns[key] = value
            except Exception as e:
                logger.error(f"Lỗi khi load semantic patterns: {str(e)}")
        else:
            if file_path and os.path.dirname(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(default_patterns, f, ensure_ascii=False, indent=2)
        
        return default_patterns
    
    def _load_context_rules(self, file_path: str) -> Dict[str, List[Dict]]:
        """Load context rules từ file hoặc sử dụng mặc định"""
        default_rules = {
            "time_context": [
                {"keywords": ["giờ", "phút"], "intent": "set-alarm", "confidence_boost": 0.1},
                {"keywords": ["sáng", "chiều", "tối", "đêm"], "intent": "set-alarm", "confidence_boost": 0.1}
            ],
            "person_context": [
                {"keywords": ["mẹ", "bố", "con", "cháu"], "intent": "call", "confidence_boost": 0.2},
                {"keywords": ["bạn", "anh", "chị"], "intent": "general-conversation", "confidence_boost": 0.1}
            ],
            "action_context": [
                {"keywords": ["ngủ", "đi"], "intent": "set-alarm", "confidence_boost": 0.15}
            ],
            "multi_turn_context": [
                {"previous_intent": "set-alarm", "keywords": ["giờ", "phút", "sáng"], "intent": "set-alarm", "confidence_boost": 0.3},
                {"previous_intent": "call", "keywords": ["số", "gọi", "điện"], "intent": "call", "confidence_boost": 0.3},
                {"previous_intent": "check-weather", "keywords": ["mưa", "nắng", "mây"], "intent": "check-weather", "confidence_boost": 0.3}
            ],
            "intent_disambiguation": [
                {"ambiguous_intents": ["call", "send-mess"], "keywords": ["gọi", "điện"], "intent": "call", "confidence_boost": 0.25},
                {"ambiguous_intents": ["call", "send-mess"], "keywords": ["nhắn", "tin"], "intent": "send-mess", "confidence_boost": 0.25}
            ]
        }
        
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_rules = json.load(f)
                    for key, value in loaded_rules.items():
                        if isinstance(value, list):
                            valid_rules = []
                            for rule in value:
                                if isinstance(rule, dict):
                                    valid_rules.append(rule)
                            default_rules[key] = valid_rules
            except Exception as e:
                logger.error(f"Lỗi khi load context rules: {str(e)}")
        else:
            if file_path and os.path.dirname(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(default_rules, f, ensure_ascii=False, indent=2)
        
        return default_rules
    
    def _load_intent_fallback(self, file_path: str) -> Dict[str, Any]:
        """Load intent fallback từ file hoặc sử dụng mặc định"""
        default_fallback = {
            "confidence_thresholds": {
                "high": 0.8,
                "medium": 0.6,
                "low": 0.4,
                "very_low": 0.2
            },
            "fallback_intents": {
                "default": "unknown",
                "clarification": "help"
            },
            "intent_suggestions": {
                "call": ["Bạn muốn gọi cho ai?", "Bạn cần gọi điện thoại phải không?"],
                "set-alarm": ["Bạn muốn đặt báo thức lúc mấy giờ?", "Bạn cần đặt báo thức phải không?"],
                "send-mess": ["Bạn muốn nhắn tin cho ai?", "Bạn cần gửi tin nhắn phải không?"],
                "check-weather": ["Bạn muốn xem thời tiết ở đâu?", "Bạn cần biết thời tiết phải không?"],
                "read-news": ["Bạn muốn đọc tin tức về chủ đề gì?", "Bạn cần đọc tin tức phải không?"],
                "check-health-status": ["Bạn muốn kiểm tra chỉ số sức khỏe nào?", "Bạn cần theo dõi sức khỏe phải không?"],
                "unknown": ["Xin lỗi, tôi chưa hiểu bạn muốn gì. Bạn có thể nói rõ hơn được không?", 
                            "Tôi chưa hiểu ý bạn. Bạn có thể diễn đạt khác được không?"]
            }
        }
        
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_fallback = json.load(f)
                    for key, value in loaded_fallback.items():
                        if key in default_fallback and isinstance(value, dict):
                            default_fallback[key].update(value)
                        else:
                            default_fallback[key] = value
            except Exception as e:
                logger.error(f"Lỗi khi load intent fallback: {str(e)}")
        else:
            if file_path and os.path.dirname(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(default_fallback, f, ensure_ascii=False, indent=2)
        
        return default_fallback
    
    def _initialize_vector_store(self) -> None:
        """Khởi tạo vector store với intent synonyms"""
        if not self.config.get("enable_vectorstore", True):
            return
            
        if self.vector_store is None:
            return
        all_texts = []
        all_intent_mapping = {}
        
        try:
            for intent, synonyms in self.knowledge_base["intent_synonyms"].items():
                for synonym in synonyms:
                    all_texts.append(synonym)
                    all_intent_mapping[synonym] = intent
            
            embeddings = []
            batch_size = 16
            for i in range(0, len(all_texts), batch_size):
                batch = all_texts[i:i+batch_size]
                batch_embeddings = self._batch_encode_texts(batch)
                embeddings.extend(batch_embeddings)
            
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.vector_store.add_vectors(all_texts, embeddings_array)
        except Exception as e:
            logger.warning(f"Vector store initialization failed: {e}")
    
    def _batch_encode_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Encode một batch các texts thành embeddings"""
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                cached_embedding = self.cache.get_embedding(text)
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                else:
                    embedding = self.get_text_embedding(text)
                    self.cache.set_embedding(text, embedding)
                    embeddings.append(embedding)
        
        return embeddings
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Lấy embedding của text sử dụng PhoBERT"""
        if self.model is None or self.tokenizer is None:
            return np.zeros(768, dtype=np.float32)
            
        cached_embedding = self.cache.get_embedding(text)
        if cached_embedding is not None and self.config.get("enable_cache", True):
            return cached_embedding
        
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.config.get("max_length", 128)
            )
            
            outputs = self.model(**inputs)
            pooling_strategy = self.config.get("pooling_strategy", "mean")
            
            if pooling_strategy == "cls":
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
            elif pooling_strategy == "mean":
                attention_mask = inputs["attention_mask"]
                mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                embedding = (sum_embeddings / sum_mask).numpy()
            elif pooling_strategy == "max":
                attention_mask = inputs["attention_mask"]
                mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                embeddings = outputs.last_hidden_state * mask
                embedding = torch.max(embeddings, dim=1)[0].numpy()
            else:
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
            
            result = embedding.flatten()
            
            if self.config.get("enable_cache", True):
                self.cache.set_embedding(text, result)
            
            return result
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Tính semantic similarity giữa 2 text"""
        if self.model is None or self.tokenizer is None:
            return 0.0
            
        cache_key = (text1, text2)
        reverse_cache_key = (text2, text1)
        
        cached_similarity = self.cache.get_similarity(cache_key)
        if cached_similarity is not None and self.config.get("enable_cache", True):
            return cached_similarity
        
        cached_similarity = self.cache.get_similarity(reverse_cache_key)
        if cached_similarity is not None and self.config.get("enable_cache", True):
            return cached_similarity
        
        try:
            emb1 = self.get_text_embedding(text1)
            emb2 = self.get_text_embedding(text2)
            
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            
            if self.config.get("enable_cache", True):
                self.cache.set_similarity(cache_key, similarity)
            
            return similarity
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def find_similar_intents(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Tìm các intent tương tự dựa trên semantic similarity"""
        if self.vector_store is None:
            return [("call", 0.0)]
            
        cached_result = self.cache.get_result(text)
        if cached_result is not None and self.config.get("enable_cache", True):
            if "semantic_similarity" in cached_result:
                semantic_results = [(k, v) for k, v in cached_result["semantic_similarity"].items()]
                semantic_results.sort(key=lambda x: x[1], reverse=True)
                return semantic_results[:top_k]
        
        if self.config.get("enable_vectorstore", True):
            try:
                text_embedding = self.get_text_embedding(text)
                similar_texts = self.vector_store.search(text_embedding, top_k * 2)
                
                intent_scores = defaultdict(float)
                for synonym, score in similar_texts:
                    for intent, synonyms in self.knowledge_base["intent_synonyms"].items():
                        if synonym in synonyms:
                            intent_scores[intent] = max(intent_scores[intent], score)
                
                similarities = [(intent, score) for intent, score in intent_scores.items()]
            except Exception as e:
                logger.warning(f"Vector store search failed: {e}")
                return [("call", 0.0)]
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
        else:
            similarities = []
            
            for intent, synonyms in self.knowledge_base["intent_synonyms"].items():
                max_similarity = 0
                for synonym in synonyms:
                    similarity = self.calculate_semantic_similarity(text, synonym)
                    max_similarity = max(max_similarity, similarity)
                
                similarities.append((intent, max_similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
    
    def extract_context_features(self, text: str) -> Dict[str, Any]:
        """Trích xuất context features từ text"""
        text_lower = text.lower()
        features = {
            "has_time": False,
            "has_person": False,
            "has_action": False,
            "has_object": False,
            "has_location": False,
            "time_keywords": [],
            "person_keywords": [],
            "action_keywords": [],
            "object_keywords": [],
            "location_keywords": []
        }
        
        if self.config.get("enable_fuzzy_matching", True):
            for category, keywords in self.knowledge_base["context_keywords"].items():
                fuzzy_matches = self.fuzzy_matcher.contains_fuzzy(text_lower, keywords)
                found_keywords = [match[0] for match in fuzzy_matches]
                
                if found_keywords:
                    features[f"has_{category}"] = True
                    features[f"{category}_keywords"] = found_keywords
        else:
            for category, keywords in self.knowledge_base["context_keywords"].items():
                found_keywords = [kw for kw in keywords if kw in text_lower]
                if found_keywords:
                    features[f"has_{category}"] = True
                    features[f"{category}_keywords"] = found_keywords
        
        if hasattr(self.entity_extractor, 'extract_all_entities'):
            entities = self.entity_extractor.extract_all_entities(text, "call")  # type: ignore
        else:
            entities = getattr(self.entity_extractor, 'extract_entities', lambda x: {})(text)  # type: ignore
        
        for entity_type, values in entities.items():
            if values:
                features[f"has_{entity_type}"] = True
                features[f"{entity_type}_entities"] = values
        
        context = self.conversation_context.get_current_context()
        if context["current_intent"]:
            features["previous_intent"] = context["current_intent"]
        
        if context["current_entities"]:
            features["previous_entities"] = context["current_entities"]
        
        return features
    
    def apply_context_rules(self, text: str, base_intent: str, base_confidence: float, 
                           context_features: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
        """Áp dụng context rules để điều chỉnh intent và confidence"""
        if context_features is None:
            context_features = self.extract_context_features(text)
        
        adjusted_intent = base_intent
        adjusted_confidence = base_confidence
        
        for rule_category, rules in self.context_rules.items():
            if rule_category == "multi_turn_context" and "previous_intent" not in context_features:
                continue
                
            for rule in rules:
                if not isinstance(rule, dict):
                    logger.warning(f"Rule is not a dict: {type(rule)} - {rule}")
                    continue
                
                if rule_category == "multi_turn_context":
                    if context_features.get("previous_intent") == rule.get("previous_intent"):
                        keywords = rule.get("keywords", [])
                        if any(kw in text.lower() for kw in keywords):
                            adjusted_intent = rule.get("intent", adjusted_intent)
                            adjusted_confidence += rule.get("confidence_boost", 0)
                            break
                
                elif rule_category == "intent_disambiguation":
                    ambiguous_intents = rule.get("ambiguous_intents", [])
                    if base_intent in ambiguous_intents:
                        keywords = rule.get("keywords", [])
                        if any(kw in text.lower() for kw in keywords):
                            intent_to_use = rule.get("intent")
                            if intent_to_use:
                                adjusted_intent = intent_to_use
                                adjusted_confidence += rule.get("confidence_boost", 0)
                            break
                
                else:
                    keywords = rule.get("keywords", [])
                    required_keywords = rule.get("required_keywords", [])
                    
                    has_required = True
                    if required_keywords:
                        has_required = any(rk in text.lower() for rk in required_keywords)
                    
                    if has_required:
                        if self.config.get("enable_fuzzy_matching", True):
                            fuzzy_matches = self.fuzzy_matcher.contains_fuzzy(text.lower(), keywords)
                            has_match = len(fuzzy_matches) > 0
                        else:
                            has_match = any(kw in text.lower() for kw in keywords)
                        
                        if has_match:
                            rule_intent = rule.get("intent")
                            confidence_boost = rule.get("confidence_boost", 0)
                            
                            if base_intent != rule_intent:
                                base_synonyms = " ".join(self.knowledge_base["intent_synonyms"].get(base_intent, []))
                                rule_synonyms = " ".join(self.knowledge_base["intent_synonyms"].get(rule_intent, []))
                                
                                base_similarity = self.calculate_semantic_similarity(base_synonyms, rule_synonyms)
                                
                                if base_similarity < 0.5:
                                    adjusted_intent = rule_intent
                                    adjusted_confidence = base_confidence + confidence_boost
                            else:
                                adjusted_confidence += confidence_boost
        
        if context_features.get("has_time") and adjusted_intent in ["set-alarm"]:
            adjusted_confidence += 0.1
        
        if context_features.get("has_person") and adjusted_intent in ["call", "send-mess"]:
            adjusted_confidence += 0.1
        
        if "video call" in text.lower() or "video" in text.lower():
            if adjusted_intent == "call":
                adjusted_intent = "make-video-call"
                adjusted_confidence += 0.2
        
        adjusted_confidence = min(adjusted_confidence, 1.0)
        
        if adjusted_intent is None:
            adjusted_intent = "unknown"
        
        return str(adjusted_intent), adjusted_confidence
    
    def pattern_matching(self, text: str) -> List[Tuple[str, float]]:
        """Pattern matching để nhận diện intent"""
        text_lower = text.lower()
        pattern_scores = []
        
        for pattern_type, patterns in self.semantic_patterns.items():
            intent = pattern_type.split("_")[0]
            
            if pattern_type.endswith("_patterns"):
                intent = pattern_type[:-9]
            
            intent_mapping = {
                "alarm": "set-alarm",
                "message": "send-mess",
                "weather": "get-info",
                "news": "get-info",
                "health": "get-info",
                "conversation": "unknown"
            }
            
            if intent in intent_mapping:
                intent = intent_mapping[intent]
            
            if intent == "send-mess":
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        pattern_scores.append((intent, 0.8))
                        break
                continue
            
            max_score = 0
            for pattern in patterns:
                if self.config.get("enable_fuzzy_matching", True):
                    matches = re.findall(pattern, text_lower)
                    if matches:
                        for match in matches:
                            match_text = match[0] if isinstance(match, tuple) else match
                            score = len(match_text) / len(text_lower) * 0.5
                            if match_text in text_lower[:len(text_lower)//2]:  # Match ở đầu câu
                                score += 0.3
                            max_score = max(max_score, score)
                    
                    try:
                        plain_pattern = pattern.replace(r".*", " ").replace(r"(?:", "").replace(")", "")
                        plain_pattern = re.sub(r"[\(\)\[\]\{\}\?\*\+\|\\]", "", plain_pattern).strip()
                        
                        if plain_pattern and len(plain_pattern) > 3:
                            fuzzy_ratio = fuzz.token_set_ratio(plain_pattern, text_lower)
                            fuzzy_score = fuzzy_ratio / 100.0 * 0.4
                            max_score = max(max_score, fuzzy_score)
                    except:
                        pass
                else:
                    matches = re.findall(pattern, text_lower)
                    if matches:
                        for match in matches:
                            match_text = match[0] if isinstance(match, tuple) else match
                            score = len(match_text) / len(text_lower) * 0.5
                            if match_text in text_lower[:len(text_lower)//2]:  # Match ở đầu câu
                                score += 0.3
                            max_score = max(max_score, score)
            
            if max_score > 0:
                pattern_scores.append((intent, max_score))
        
        return pattern_scores
    
    def keyword_matching(self, text: str) -> List[Tuple[str, float]]:
        """Keyword matching để nhận diện intent"""
        text_lower = text.lower()
        keyword_scores = []
        
        for multi_intent, indicators in self.knowledge_base.get("multi_intent_indicators", {}).items():
            score = 0
            matched_indicators = []
            
            for indicator in indicators:
                if self.config.get("enable_fuzzy_matching", True):
                    fuzzy_matches = self.fuzzy_matcher.contains_fuzzy(text_lower, [indicator])
                    if fuzzy_matches:
                        keyword, match_score = fuzzy_matches[0]
                        score += 0.2 * (match_score / 100)
                        matched_indicators.append(keyword)
                else:
                    if indicator in text_lower:
                        score += 0.2
                        matched_indicators.append(indicator)
            
            if score > 0:
                intents = multi_intent.split(",")
                best_intent = intents[0]
                best_intent_score = 0
                
                for intent in intents:
                    intent_indicators = self.knowledge_base["intent_indicators"].get(intent, [])
                    intent_score = 0
                    
                    for indicator in intent_indicators:
                        if indicator in text_lower:
                            intent_score += 0.1
                    
                    if intent_score > best_intent_score:
                        best_intent_score = intent_score
                        best_intent = intent
                
                keyword_scores.append((best_intent, min(score + best_intent_score, 1.0)))
        
        for intent, indicators in self.knowledge_base["intent_indicators"].items():
            score = 0
            matched_indicators = []
            
            if intent == "send-mess":
                message_keywords = ["nhắn", "tin", "gửi", "soạn", "tin nhắn", "nhắn tin"]
                for keyword in message_keywords:
                    if keyword in text_lower:
                        score += 0.3
                        matched_indicators.append(keyword)
                        
                        if text_lower.startswith(keyword) or text_lower.find(f" {keyword}") < len(text_lower) // 3:
                            score += 0.2
                
                if score > 0:
                    keyword_scores.append((intent, min(score, 1.0)))
                continue
            
            if self.config.get("enable_fuzzy_matching", True):
                fuzzy_matches = self.fuzzy_matcher.contains_fuzzy(text_lower, indicators)
                for keyword, match_score in fuzzy_matches:
                    score += 0.2 * (match_score / 100)
                    matched_indicators.append(keyword)
                    
                    if f" {keyword} " in f" {text_lower} ":
                        score += 0.1
            else:
                for indicator in indicators:
                    if indicator in text_lower:
                        score += 0.2
                        matched_indicators.append(indicator)
                        
                        if f" {indicator} " in f" {text_lower} ":
                            score += 0.1
            
            if score > 0:
                for indicator in matched_indicators:
                    if text_lower.startswith(indicator) or text_lower.find(f" {indicator}") < len(text_lower) // 3:
                        score += 0.1
                        break
                
                keyword_scores.append((intent, min(score, 1.0)))
        
        return keyword_scores
    
    def reasoning_predict(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Predict intent sử dụng reasoning engine"""
        start_time = time.time()
        
        cached_result = self.cache.get_result(text)
        if cached_result is not None and self.config.get("enable_cache", True):
            return cached_result
        
        if context:
            self.conversation_context.session_data.update(context)
        
        semantic_results = self.find_similar_intents(text)
        pattern_results = self.pattern_matching(text)
        keyword_results = self.keyword_matching(text)
        
        try:
            if hasattr(self.entity_extractor, 'extract_all_entities'):
                entities = self.entity_extractor.extract_all_entities(text, "call")  # type: ignore
            else:
                entities = getattr(self.entity_extractor, 'extract_entities', lambda x: {})(text)  # type: ignore
            
            if not isinstance(entities, dict):
                entities = {}
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            entities = {}
        
        context_features = self.extract_context_features(text)
        
        combined_scores = defaultdict(float)
        
        semantic_weight = self.config.get("semantic_weight", 0.15)
        pattern_weight = self.config.get("pattern_weight", 0.55)
        keyword_weight = self.config.get("keyword_weight", 0.30)
        
        for intent, score in semantic_results:
            combined_scores[intent] += score * semantic_weight
        
        for intent, score in pattern_results:
            combined_scores[intent] += score * pattern_weight
        
        for intent, score in keyword_results:
            combined_scores[intent] += score * keyword_weight
        
        if combined_scores:
            best_intent = max(combined_scores.items(), key=lambda x: x[1])
            base_intent, base_confidence = best_intent
            
            adjusted_intent, adjusted_confidence = self.apply_context_rules(
                text, base_intent, base_confidence, context_features
            )
            
            adjusted_intent = self.normalize_intent.get(adjusted_intent, adjusted_intent)
            
            entities_normalized: Dict[str, List[str]] = {}
            if isinstance(entities, dict):
                for key, value in entities.items():
                    if isinstance(value, str):
                        entities_normalized[key] = [value]
                    elif isinstance(value, list):
                        entities_normalized[key] = [str(v) for v in value]
                    else:
                        entities_normalized[key] = [str(value)]
            
            validation_result = self.validate_reasoning_result(
                text, adjusted_intent, adjusted_confidence, entities_normalized if entities_normalized else None
            )
            
            fallback_result = self.apply_fallback_strategy(
                text, validation_result, semantic_results
            )
            
            result = {
                "intent": fallback_result["intent"],
                "confidence": fallback_result["confidence"],
                "reasoning_method": "semantic_pattern_keyword_context",
                "semantic_similarity": dict(semantic_results),
                "pattern_matching": dict(pattern_results),
                "keyword_matching": dict(keyword_results),
                "context_features": context_features,
                "entities": entities,
                "validation": validation_result,
                "fallback": fallback_result.get("fallback_info"),
                "explanation": self.generate_explanation(text, fallback_result["intent"], validation_result, entities_normalized if entities_normalized else None),
                "suggestions": fallback_result.get("suggestions", []),
                "processing_time": time.time() - start_time
            }
            
            if self.config.get("enable_cache", True):
                self.cache.set_result(text, result)
            
            self.conversation_context.add_turn(text, result)
            
            return result
        else:
            unknown_result = {
                "intent": "unknown",
                "confidence": 0.0,
                "reasoning_method": "no_match",
                "explanation": "Không tìm thấy pattern nào phù hợp",
                "suggestions": self.intent_fallback["intent_suggestions"]["unknown"],
                "processing_time": time.time() - start_time
            }
            
            if self.config.get("enable_cache", True):
                self.cache.set_result(text, unknown_result)
            
            self.conversation_context.add_turn(text, unknown_result)
            
            return unknown_result
    
    def validate_reasoning_result(self, text: str, intent: str, confidence: float, 
                                 entities: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """Validate kết quả reasoning"""
        validation = {
            "intent": intent,
            "confidence": confidence,
            "is_valid": True,
            "warnings": [],
            "suggestions": []
        }
        
        if confidence < self.similarity_threshold:
            validation["is_valid"] = False
            validation["warnings"].append(f"Confidence quá thấp: {confidence:.3f}")
            validation["suggestions"].append("Cần xác nhận lại từ người dùng")
        
        text_lower = text.lower()
        conflicting_patterns = {
            "call": ["nhắn tin", "gửi tin nhắn", "soạn tin"],
            "send-mess": ["gọi", "điện thoại", "alo"],
            "set-alarm": ["gửi tin nhắn", "gọi", "nhắn tin"]
        }
        
        if intent in conflicting_patterns:
            conflicting = conflicting_patterns[intent]
            if self.config.get("enable_fuzzy_matching", True):
                fuzzy_matches = self.fuzzy_matcher.contains_fuzzy(text_lower, conflicting)
                if fuzzy_matches:
                    validation["confidence"] *= 0.7
                    validation["warnings"].append("Có từ khóa mâu thuẫn với intent")
            else:
                if any(pattern in text_lower for pattern in conflicting):
                    validation["confidence"] *= 0.7
                    validation["warnings"].append("Có từ khóa mâu thuẫn với intent")
        
        intent_entity_requirements = {
            "call": ["person"],
            "send-mess": ["person"],
            "set-alarm": ["time"],
            "get-info": [],
            "search-internet": [], 
            "search-youtube": [],
            "make-video-call": ["person"],
            "open-cam": [], 
            "control-device": [],
            "add-contacts": ["person"]
        }
        
        if intent in intent_entity_requirements and entities:
            required_entities = intent_entity_requirements[intent]
            missing_entities = []
            
            for entity_type in required_entities:
                if entity_type not in entities or not entities[entity_type]:
                    missing_entities.append(entity_type)
            
            if missing_entities:
                validation["confidence"] *= 0.8
                validation["warnings"].append(f"Thiếu thông tin {', '.join(missing_entities)} cho intent {intent}")
                
                for entity_type in missing_entities:
                    if entity_type == "time":
                        validation["suggestions"].append("Bạn muốn thực hiện lúc mấy giờ?")
                    elif entity_type == "person":
                        validation["suggestions"].append("Bạn muốn thực hiện với ai?")
                    elif entity_type == "location":
                        validation["suggestions"].append("Bạn muốn thực hiện ở đâu?")
                    elif entity_type == "action":
                        validation["suggestions"].append("Bạn muốn làm gì?")
                    elif entity_type == "object":
                        validation["suggestions"].append("Bạn muốn làm với cái gì?")
        
        current_context = self.conversation_context.get_current_context()
        if current_context["current_intent"] and current_context["current_intent"] != intent:
            if intent in intent_entity_requirements and entities:
                required_entities = intent_entity_requirements[intent]
                for entity_type in required_entities:
                    if (entity_type not in entities or not entities[entity_type]) and \
                       entity_type in current_context["current_entities"] and \
                       current_context["current_entities"][entity_type]:
                        validation["confidence"] *= 1.1
                        validation["warnings"].append(f"Sử dụng {entity_type} từ context trước đó")
        
        return validation
    
    def apply_fallback_strategy(self, text: str, validation: Dict[str, Any], 
                              semantic_results: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Áp dụng fallback strategy dựa trên confidence và validation"""
        if not isinstance(validation, dict):
            logger.warning(f"validation is not a dict: {type(validation)} - {validation}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "suggestions": []
            }
        
        intent = validation.get("intent", "unknown")
        confidence = validation.get("confidence", 0.0)
        
        fallback_result = {
            "intent": intent,
            "confidence": confidence
        }
        
        thresholds = self.intent_fallback["confidence_thresholds"]
        
        if confidence < thresholds["very_low"]:
            fallback_result["intent"] = self.intent_fallback["fallback_intents"]["default"]
            fallback_result["confidence"] = 0.2
            fallback_result["fallback_info"] = {
                "original_intent": intent,
                "original_confidence": confidence,
                "reason": "confidence_too_low"
            }
            fallback_result["suggestions"] = self.intent_fallback["intent_suggestions"]["unknown"]
        
        elif confidence < thresholds["low"] and validation.get("warnings"):
            fallback_result["intent"] = self.intent_fallback["fallback_intents"]["clarification"]
            fallback_result["confidence"] = 0.4
            fallback_result["fallback_info"] = {
                "original_intent": intent,
                "original_confidence": confidence,
                "reason": "low_confidence_with_warnings",
                "warnings": validation.get("warnings", [])
            }
            
            if intent in self.intent_fallback["intent_suggestions"]:
                fallback_result["suggestions"] = self.intent_fallback["intent_suggestions"][intent]
            else:
                fallback_result["suggestions"] = self.intent_fallback["intent_suggestions"]["unknown"]
        
        elif confidence < thresholds["medium"] and len(semantic_results) >= 2:
            top_intents = semantic_results[:2]
            if abs(top_intents[0][1] - top_intents[1][1]) < 0.1:
                fallback_result["intent"] = self.intent_fallback["fallback_intents"]["clarification"]
                fallback_result["confidence"] = 0.5
                fallback_result["fallback_info"] = {
                    "original_intent": intent,
                    "original_confidence": confidence,
                    "reason": "ambiguous_intents",
                    "ambiguous_intents": [i[0] for i in top_intents]
                }
                
                suggestions = []
                for ambiguous_intent, _ in top_intents:
                    if ambiguous_intent in self.intent_fallback["intent_suggestions"]:
                        suggestions.extend(self.intent_fallback["intent_suggestions"][ambiguous_intent][:1])
                
                if not suggestions:
                    suggestions = self.intent_fallback["intent_suggestions"]["unknown"]
                
                fallback_result["suggestions"] = suggestions
        
        elif validation.get("warnings") and len(validation.get("warnings", [])) > 1:
            fallback_result["suggestions"] = validation.get("suggestions", [])
            
            if not fallback_result["suggestions"] and intent in self.intent_fallback["intent_suggestions"]:
                fallback_result["suggestions"] = self.intent_fallback["intent_suggestions"][intent]
        
        else:
            if validation.get("suggestions"):
                fallback_result["suggestions"] = validation.get("suggestions")
            elif intent in self.intent_fallback["intent_suggestions"]:
                fallback_result["suggestions"] = [self.intent_fallback["intent_suggestions"][intent][0]] \
                                               if self.intent_fallback["intent_suggestions"][intent] else []
        
        return fallback_result
    
    def generate_explanation(self, text: str, intent: str, validation: Dict, 
                           entities: Optional[Dict[str, List[str]]] = None) -> str:
        """Tạo explanation cho kết quả reasoning"""
        explanation_parts = []
        
        if isinstance(validation, dict) and 'confidence' in validation:
            confidence = validation['confidence']
            if isinstance(confidence, (int, float)):
                explanation_parts.append(f"Intent '{intent}' được chọn với confidence {confidence:.3f}")
            else:
                explanation_parts.append(f"Intent '{intent}' được chọn với confidence {confidence}")
        else:
            explanation_parts.append(f"Intent '{intent}' được chọn")
        
        if entities and isinstance(entities, dict):
            entity_explanations = []
            for entity_type, values in entities.items():
                if values:
                    if isinstance(values, list):
                        values_str = [str(v) for v in values]
                        entity_explanations.append(f"{entity_type}: {', '.join(values_str)}")
                    else:
                        entity_explanations.append(f"{entity_type}: {str(values)}")
            
            if entity_explanations:
                explanation_parts.append(f"Entities: {'; '.join(entity_explanations)}")
        
        if isinstance(validation, dict) and validation.get("warnings"):
            warnings = validation.get("warnings", [])
            if isinstance(warnings, list):
                warnings_str = [str(w) for w in warnings]
                explanation_parts.append(f"Cảnh báo: {', '.join(warnings_str)}")
        
        if isinstance(validation, dict):
            context_features = validation.get("context_features", {})
            if isinstance(context_features, dict):
                context_explanations = []
                if context_features.get("has_time"):
                    context_explanations.append("có thông tin thời gian")
                if context_features.get("has_person"):
                    context_explanations.append("có thông tin người")
                if context_features.get("has_action"):
                    context_explanations.append("có hành động cụ thể")
                if context_features.get("has_location"):
                    context_explanations.append("có thông tin địa điểm")
                
                if context_explanations:
                    explanation_parts.append(f"Context: {', '.join(context_explanations)}")
        
        current_context = self.conversation_context.get_current_context()
        if isinstance(current_context, dict) and current_context.get("current_intent"):
            explanation_parts.append(f"Previous intent: {current_context['current_intent']}")
        
        return ". ".join(explanation_parts)
    
    def batch_reasoning(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Reasoning cho nhiều text cùng lúc"""
        results = []
        for text in texts:
            result = self.reasoning_predict(text)
            results.append(result)
        return results
    
    def update_knowledge_base(self, new_patterns: Dict[str, List[str]]):
        """Cập nhật knowledge base với patterns mới"""
        for intent, patterns in new_patterns.items():
            if intent in self.knowledge_base["intent_synonyms"]:
                self.knowledge_base["intent_synonyms"][intent].extend(patterns)
            else:
                self.knowledge_base["intent_synonyms"][intent] = patterns
        
        if self.config.get("enable_vectorstore", True):
            self._initialize_vector_store()
    
    def save_knowledge_base(self, file_path: Optional[str] = None):
        """Lưu knowledge base"""
        if file_path is None:
            file_path = self.config.get("knowledge_base_path", "knowledge_base.json")
        if file_path is None:
            file_path = "knowledge_base.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
    
    def load_knowledge_base(self, file_path: Optional[str] = None):
        """Load knowledge base"""
        if file_path is None:
            file_path = self.config.get("knowledge_base_path", "knowledge_base.json")
        if file_path is None:
            file_path = "knowledge_base.json"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)
        
        if self.config.get("enable_vectorstore", True):
            self._initialize_vector_store()
    
    def reset_conversation_context(self):
        """Reset conversation context"""
        self.conversation_context.reset()
    
    def clear_cache(self):
        """Clear cache"""
        self.cache.clear()