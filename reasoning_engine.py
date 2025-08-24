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
from config import model_config, intent_config

# Thiết lập logging
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
            # Xóa các item ít sử dụng nhất
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
        return results
    
    def contains_fuzzy(self, text: str, keywords: List[str], threshold: Optional[int] = None) -> List[Tuple[str, int]]:
        """Kiểm tra xem text có chứa bất kỳ keyword nào không (fuzzy matching)"""
        match_threshold = threshold if threshold is not None else self.threshold
        matches = []
        
        for keyword in keywords:
            # Tìm tất cả các substring có độ dài tương tự keyword
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
    
    def __init__(self, vector_dim: int = 768):
        self.vector_dim = vector_dim
        self.index = faiss.IndexFlatIP(vector_dim)  # Inner product để cosine similarity
        self.text_mapping = []  # Lưu text tương ứng với các vector
        
    def add_vectors(self, texts: List[str], vectors: np.ndarray) -> None:
        """Thêm vectors vào index"""
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.text_mapping.extend(texts)
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Tìm kiếm vectors gần nhất"""
        # Normalize query vector
        query_vector = query_vector.reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.text_mapping):  # Check valid index
                results.append((self.text_mapping[idx], float(scores[0][i])))
        
        return results
    
    def reset(self) -> None:
        """Reset index"""
        self.index = faiss.IndexFlatIP(self.vector_dim)
        self.text_mapping = []

class ConversationContext:
    """Quản lý ngữ cảnh hội thoại"""
    
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
        
        # Cập nhật intent và entities hiện tại
        if system_response.get("intent") and system_response.get("confidence", 0) > 0.5:
            self.current_intent = system_response["intent"]
        
        if system_response.get("entities"):
            self.current_entities.update(system_response["entities"])
    
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
        # Mẫu cấu trúc, có thể được load từ file
        return {
            "time": {
                "patterns": [
                    r"(\d{1,2})[h:]\s*(\d{1,2})?",  # 8h30, 8:30
                    r"(\d{1,2})\s*(giờ|tiếng)\s*(\d{1,2})?\s*(phút)?",  # 8 giờ 30 phút
                    r"(sáng|trưa|chiều|tối|đêm)",  # sáng, chiều, tối
                    r"(hôm nay|ngày mai|hôm qua)",  # hôm nay, ngày mai
                ],
                "keywords": ["giờ", "phút", "sáng", "trưa", "chiều", "tối", "đêm", 
                             "hôm nay", "ngày mai", "hôm qua", "tuần", "tháng"]
            },
            "person": {
                "patterns": [
                    r"(mẹ|ba|bố|bạn|anh|chị|em|cô|chú|bác|ông|bà)\s*(của|tôi|tui|mình)?",
                ],
                "keywords": ["mẹ", "ba", "bố", "bạn", "anh", "chị", "em", "cô", "chú", "bác", "ông", "bà"]
            },
            "location": {
                "patterns": [
                    r"(tại|ở)\s*(.*?)(nhé|nha|nhá|\.|\?|$)",  # tại/ở + location
                    r"(nhà|công ty|văn phòng|bệnh viện|trường|phòng)",
                ],
                "keywords": ["nhà", "công ty", "văn phòng", "bệnh viện", "trường", "phòng", "quán", "tại", "ở"]
            }
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Trích xuất entities từ text"""
        entities = {entity_type: [] for entity_type in self.entity_patterns}
        
        # Pattern matching
        for entity_type, patterns_data in self.entity_patterns.items():
            # Regex pattern matching
            for pattern in patterns_data["patterns"]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple):  # Group captures
                            # Nối các group lại với nhau
                            entity_value = " ".join([m for m in match if m])
                            if entity_value:
                                entities[entity_type].append(entity_value.strip())
                        else:
                            entities[entity_type].append(match.strip())
            
            # Fuzzy keyword matching
            fuzzy_matches = self.fuzzy_matcher.contains_fuzzy(text, patterns_data["keywords"])
            for keyword, _ in fuzzy_matches:
                # Trích xuất phrase chứa keyword
                keyword_idx = text.lower().find(keyword.lower())
                if keyword_idx >= 0:
                    start_idx = max(0, text.rfind(" ", 0, keyword_idx) + 1)
                    end_idx = text.find(" ", keyword_idx + len(keyword))
                    if end_idx == -1:
                        end_idx = len(text)
                    
                    phrase = text[start_idx:end_idx].strip()
                    if phrase and phrase not in entities[entity_type]:
                        entities[entity_type].append(phrase)
        
        # Deduplicate and clean entities
        for entity_type in entities:
            entities[entity_type] = list(set(entities[entity_type]))
        
        return entities

class ReasoningEngine:
    """Hệ thống tự suy luận sử dụng PhoBERT"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Khởi tạo cache
        self.cache = ReasoningCache(max_size=2000)
        
        # Đọc config từ file nếu có
        self.config = self._load_config(config_path)
        
        # Khởi tạo tokenizer và model với safetensors
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.get("model_name", model_config.model_name))
        self.model = AutoModel.from_pretrained(
            self.config.get("model_name", model_config.model_name),
            use_safetensors=True,
            trust_remote_code=True
        )
        self.model.eval()
        
        # Khởi tạo fuzzy matcher
        self.fuzzy_matcher = FuzzyMatcher(threshold=self.config.get("fuzzy_threshold", 75))
        
        # Khởi tạo vector store
        self.vector_store = VectorStore(vector_dim=self.model.config.hidden_size)
        
        # Khởi tạo entity extractor
        self.entity_extractor = EntityExtractor(self.fuzzy_matcher)
        
        # Khởi tạo conversation context
        self.conversation_context = ConversationContext(max_history=self.config.get("max_history", 5))
        
        # Khởi tạo knowledge base
        self.knowledge_base = self._load_knowledge_base(
            self.config.get("knowledge_base_path", "knowledge_base.json")
        )
        
        # Khởi tạo semantic patterns
        self.semantic_patterns = self._load_semantic_patterns(
            self.config.get("patterns_path", "semantic_patterns.json")
        )
        
        # Khởi tạo context rules
        self.context_rules = self._load_context_rules(
            self.config.get("rules_path", "context_rules.json")
        )
        
        # Khởi tạo intent fallback config
        self.intent_fallback = self._load_intent_fallback(
            self.config.get("fallback_path", "intent_fallback.json")
        )
        
        # Khởi tạo vector store với intent synonyms
        self._initialize_vector_store()
        
        # Similarity threshold
        self.similarity_threshold = self.config.get("similarity_threshold", 0.6)
        
        logger.info("ReasoningEngine đã được khởi tạo thành công")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load config từ file"""
        default_config = {
            "model_name": model_config.model_name,
            "max_length": model_config.max_length,
            "similarity_threshold": 0.6,
            "fuzzy_threshold": 75,
            "max_history": 5,
            "knowledge_base_path": "knowledge_base.json",
            "patterns_path": "semantic_patterns.json",
            "rules_path": "context_rules.json",
            "fallback_path": "intent_fallback.json",
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
                "call": ["gọi", "điện thoại", "alo", "kết nối", "liên lạc", "quay số", "bấm số", "phone", "call"],
                "set-alarm": ["đặt báo thức", "nhắc nhở", "hẹn giờ", "đánh thức", "chuông báo", "ghi nhớ", "alarm", "reminder"],
                "send-mess": ["gửi tin nhắn", "nhắn tin", "text", "sms", "thông báo", "soạn tin", "message", "send"],
                "set-reminder": ["đặt nhắc nhở", "ghi nhớ", "lời nhắc", "nhắc tôi", "tạo lời nhắc", "reminder"],
                "check-weather": ["thời tiết", "nhiệt độ", "mưa", "nắng", "dự báo thời tiết", "weather", "temperature"],
                "play-media": ["phát nhạc", "bật nhạc", "nghe nhạc", "video", "mở nhạc", "chơi nhạc", "play", "music"],
                "read-news": ["đọc tin tức", "tin tức", "báo", "thời sự", "cập nhật tin", "news", "read"],
                "check-health-status": ["kiểm tra sức khỏe", "đo", "theo dõi", "chỉ số", "tình trạng", "health", "check"],
                "general-conversation": ["xin chào", "tạm biệt", "cảm ơn", "trò chuyện", "nói chuyện", "hello", "conversation"],
                "help": ["giúp đỡ", "trợ giúp", "hướng dẫn", "hỗ trợ", "không hiểu", "help", "support"],
                "unknown": ["không hiểu", "không rõ", "lạ", "không biết", "chưa rõ"]
            },
            "context_keywords": {
                "time": ["giờ", "phút", "sáng", "chiều", "tối", "mai", "hôm nay", "tuần", "tháng"],
                "person": ["mẹ", "bố", "con", "cháu", "bạn", "anh", "chị", "em", "ông", "bà"],
                "location": ["nhà", "bệnh viện", "phòng", "ngoài", "trong", "đây", "đó"],
                "action": ["uống", "ăn", "ngủ", "đi", "về", "đến", "gặp", "thăm"],
                "object": ["thuốc", "nước", "cơm", "sách", "điện thoại", "tivi", "radio"]
            },
            "intent_indicators": {
                "call": ["gọi", "điện", "phone", "call", "kết nối", "liên lạc", "cuộc gọi", "gọi thoại", "gọi điện", "thực hiện gọi", "thực hiện cuộc gọi"],
                "set-alarm": ["báo thức", "nhắc", "hẹn", "alarm", "reminder", "giờ"],
                "send-mess": ["nhắn", "tin", "message", "sms", "text", "gửi"],
                "set-reminder": ["nhắc", "nhớ", "reminder", "ghi", "lời nhắc", "uống thuốc", "thuốc", "viên thuốc"],
                "check-weather": ["thời tiết", "weather", "nhiệt", "mưa", "nắng"],
                "play-media": ["nhạc", "music", "phát", "bật", "nghe", "play"],
                "read-news": ["tin", "news", "báo", "đọc", "thời sự"],
                "check-health-status": ["sức khỏe", "health", "kiểm tra", "đo", "theo dõi"],
                "general-conversation": ["chào", "hello", "cảm ơn", "tạm biệt", "nói chuyện"],
                "help": ["giúp", "help", "hướng dẫn", "hỗ trợ"],
                "unknown": ["không hiểu", "chưa rõ", "không biết"]
            },
            "multi_intent_indicators": {
                "call,send-mess": ["gọi", "nhắn", "liên lạc"],
                "set-alarm,set-reminder": ["nhắc", "giờ", "hẹn", "đặt"],
                "check-weather,read-news": ["thời tiết", "tin tức", "cập nhật"]
            }
        }
        
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_kb = json.load(f)
                    # Merge với default knowledge base
                    for key, value in loaded_kb.items():
                        if key in default_kb:
                            if isinstance(value, dict):
                                default_kb[key].update(value)
                            else:
                                default_kb[key] = value
                        else:
                            default_kb[key] = value
                    
                    logger.info(f"Đã load knowledge base từ {file_path}")
            except Exception as e:
                logger.error(f"Lỗi khi load knowledge base: {str(e)}")
        else:
            logger.warning(f"File knowledge base không tồn tại: {file_path}. Sử dụng mặc định.")
            # Tự động tạo file nếu chưa tồn tại
            if file_path and os.path.dirname(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(default_kb, f, ensure_ascii=False, indent=2)
                logger.info(f"Đã tạo file knowledge base mặc định: {file_path}")
        
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
                r"text.*(?:cho|tới)"
            ],
            "weather_patterns": [
                r"thời tiết.*(?:hôm nay|mai|thế nào)",
                r"nhiệt độ.*(?:bao nhiêu|thế nào)",
                r"dự báo.*(?:thời tiết|mưa|nắng)"
            ],
            "media_patterns": [
                r"phát.*(?:nhạc|bài hát)",
                r"bật.*(?:nhạc|video)",
                r"nghe.*(?:nhạc|bài hát)",
                r"mở.*(?:nhạc|video)"
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
                    # Merge với default patterns
                    for key, value in loaded_patterns.items():
                        default_patterns[key] = value
                    
                    logger.info(f"Đã load semantic patterns từ {file_path}")
            except Exception as e:
                logger.error(f"Lỗi khi load semantic patterns: {str(e)}")
        else:
            logger.warning(f"File semantic patterns không tồn tại: {file_path}. Sử dụng mặc định.")
            # Tự động tạo file nếu chưa tồn tại
            if file_path and os.path.dirname(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(default_patterns, f, ensure_ascii=False, indent=2)
                logger.info(f"Đã tạo file semantic patterns mặc định: {file_path}")
        
        return default_patterns
    
    def _load_context_rules(self, file_path: str) -> Dict[str, List[Dict]]:
        """Load context rules từ file hoặc sử dụng mặc định"""
        default_rules = {
            "time_context": [
                # Chỉ boost set-alarm khi có từ khóa báo thức rõ ràng
                {"keywords": ["giờ", "phút"], "intent": "set-alarm", "confidence_boost": 0.1, "required_keywords": ["báo thức", "đánh thức"]},
                {"keywords": ["sáng", "chiều", "tối"], "intent": "set-alarm", "confidence_boost": 0.1, "required_keywords": ["báo thức", "đánh thức"]},
                {"keywords": ["mai", "hôm nay"], "intent": "set-reminder", "confidence_boost": 0.1, "required_keywords": ["nhắc", "nhớ"]}
            ],
            "person_context": [
                {"keywords": ["mẹ", "bố", "con", "cháu"], "intent": "call", "confidence_boost": 0.2},
                {"keywords": ["bạn", "anh", "chị"], "intent": "general-conversation", "confidence_boost": 0.1}
            ],
            "action_context": [
                {"keywords": ["uống thuốc", "thuốc", "viên thuốc", "kháng sinh", "tiểu đường", "huyết áp", "tim", "vitamin", "sắt", "cảm", "đau đầu"], "intent": "set-reminder", "confidence_boost": 0.3},
                {"keywords": ["uống", "ăn"], "intent": "set-reminder", "confidence_boost": 0.2},
                {"keywords": ["ngủ", "đi"], "intent": "set-alarm", "confidence_boost": 0.15}
            ],
            "multi_turn_context": [
                {"previous_intent": "set-alarm", "keywords": ["giờ", "phút", "sáng"], "intent": "set-alarm", "confidence_boost": 0.3},
                {"previous_intent": "call", "keywords": ["số", "gọi", "điện"], "intent": "call", "confidence_boost": 0.3},
                {"previous_intent": "check-weather", "keywords": ["mưa", "nắng", "mây"], "intent": "check-weather", "confidence_boost": 0.3}
            ],
            "intent_disambiguation": [
                {"ambiguous_intents": ["call", "send-mess"], "keywords": ["gọi", "điện"], "intent": "call", "confidence_boost": 0.25},
                {"ambiguous_intents": ["call", "send-mess"], "keywords": ["nhắn", "tin"], "intent": "send-mess", "confidence_boost": 0.25},
                {"ambiguous_intents": ["set-alarm", "set-reminder"], "keywords": ["báo thức", "chuông"], "intent": "set-alarm", "confidence_boost": 0.25}
            ]
        }
        
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_rules = json.load(f)
                    # Merge với default rules
                    for key, value in loaded_rules.items():
                        default_rules[key] = value
                    
                    logger.info(f"Đã load context rules từ {file_path}")
            except Exception as e:
                logger.error(f"Lỗi khi load context rules: {str(e)}")
        else:
            logger.warning(f"File context rules không tồn tại: {file_path}. Sử dụng mặc định.")
            # Tự động tạo file nếu chưa tồn tại
            if file_path and os.path.dirname(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(default_rules, f, ensure_ascii=False, indent=2)
                logger.info(f"Đã tạo file context rules mặc định: {file_path}")
        
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
                "set-reminder": ["Bạn muốn đặt nhắc nhở gì?", "Bạn cần đặt lời nhắc phải không?"],
                "check-weather": ["Bạn muốn xem thời tiết ở đâu?", "Bạn cần biết thời tiết phải không?"],
                "play-media": ["Bạn muốn nghe bài hát gì?", "Bạn cần phát nhạc phải không?"],
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
                    # Merge với default fallback
                    for key, value in loaded_fallback.items():
                        if key in default_fallback and isinstance(value, dict):
                            default_fallback[key].update(value)
                        else:
                            default_fallback[key] = value
                    
                    logger.info(f"Đã load intent fallback từ {file_path}")
            except Exception as e:
                logger.error(f"Lỗi khi load intent fallback: {str(e)}")
        else:
            logger.warning(f"File intent fallback không tồn tại: {file_path}. Sử dụng mặc định.")
            # Tự động tạo file nếu chưa tồn tại
            if file_path and os.path.dirname(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(default_fallback, f, ensure_ascii=False, indent=2)
                logger.info(f"Đã tạo file intent fallback mặc định: {file_path}")
        
        return default_fallback
    
    def _initialize_vector_store(self) -> None:
        """Khởi tạo vector store với intent synonyms"""
        if not self.config.get("enable_vectorstore", True):
            logger.info("Vector store đã bị tắt trong config")
            return
        
        logger.info("Đang khởi tạo vector store...")
        # Flatten intent synonyms
        all_texts = []
        all_intent_mapping = {}
        
        for intent, synonyms in self.knowledge_base["intent_synonyms"].items():
            for synonym in synonyms:
                all_texts.append(synonym)
                all_intent_mapping[synonym] = intent
        
        # Tính embeddings cho tất cả synonyms
        embeddings = []
        batch_size = 16
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i+batch_size]
            batch_embeddings = self._batch_encode_texts(batch)
            embeddings.extend(batch_embeddings)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Add to vector store
        self.vector_store.add_vectors(all_texts, embeddings_array)
        logger.info(f"Vector store đã được khởi tạo với {len(all_texts)} vectors")
    
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
        # Kiểm tra trong cache trước
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
            
            # Chọn pooling strategy theo config
            pooling_strategy = self.config.get("pooling_strategy", "mean")
            
            if pooling_strategy == "cls":
                # [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
            elif pooling_strategy == "mean":
                # Mean pooling
                attention_mask = inputs["attention_mask"]
                mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                embedding = (sum_embeddings / sum_mask).numpy()
            elif pooling_strategy == "max":
                # Max pooling
                attention_mask = inputs["attention_mask"]
                mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                embeddings = outputs.last_hidden_state * mask
                embedding = torch.max(embeddings, dim=1)[0].numpy()
            else:
                # Default to [CLS]
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
            
            result = embedding.flatten()
            
            # Lưu vào cache
            if self.config.get("enable_cache", True):
                self.cache.set_embedding(text, result)
            
            return result
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Tính semantic similarity giữa 2 text"""
        # Kiểm tra trong cache trước
        cache_key = (text1, text2)
        reverse_cache_key = (text2, text1)
        
        cached_similarity = self.cache.get_similarity(cache_key)
        if cached_similarity is not None and self.config.get("enable_cache", True):
            return cached_similarity
        
        cached_similarity = self.cache.get_similarity(reverse_cache_key)
        if cached_similarity is not None and self.config.get("enable_cache", True):
            return cached_similarity
        
        emb1 = self.get_text_embedding(text1)
        emb2 = self.get_text_embedding(text2)
        
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        
        # Lưu vào cache
        if self.config.get("enable_cache", True):
            self.cache.set_similarity(cache_key, similarity)
        
        return similarity
    
    def find_similar_intents(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Tìm các intent tương tự dựa trên semantic similarity"""
        # Kiểm tra trong cache trước
        cached_result = self.cache.get_result(text)
        if cached_result is not None and self.config.get("enable_cache", True):
            if "semantic_similarity" in cached_result:
                semantic_results = [(k, v) for k, v in cached_result["semantic_similarity"].items()]
                semantic_results.sort(key=lambda x: x[1], reverse=True)
                return semantic_results[:top_k]
        
        # Sử dụng vector store nếu được bật
        if self.config.get("enable_vectorstore", True):
            # Lấy embedding của text
            text_embedding = self.get_text_embedding(text)
            # Tìm kiếm trong vector store
            similar_texts = self.vector_store.search(text_embedding, top_k * 2)  # Lấy nhiều hơn để đảm bảo đủ intent
            
            # Nhóm theo intent và lấy max similarity
            intent_scores = defaultdict(float)
            for synonym, score in similar_texts:
                for intent, synonyms in self.knowledge_base["intent_synonyms"].items():
                    if synonym in synonyms:
                        intent_scores[intent] = max(intent_scores[intent], score)
            
            # Chuyển thành list và sắp xếp
            similarities = [(intent, score) for intent, score in intent_scores.items()]
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
        else:
            # Fallback sang cách cũ nếu vector store bị tắt
            similarities = []
            
            for intent, synonyms in self.knowledge_base["intent_synonyms"].items():
                max_similarity = 0
                for synonym in synonyms:
                    similarity = self.calculate_semantic_similarity(text, synonym)
                    max_similarity = max(max_similarity, similarity)
                
                similarities.append((intent, max_similarity))
            
            # Sắp xếp theo similarity giảm dần
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
        
        # Sử dụng fuzzy matching nếu được bật
        if self.config.get("enable_fuzzy_matching", True):
            # Kiểm tra các loại keywords với fuzzy matching
            for category, keywords in self.knowledge_base["context_keywords"].items():
                fuzzy_matches = self.fuzzy_matcher.contains_fuzzy(text_lower, keywords)
                found_keywords = [match[0] for match in fuzzy_matches]
                
                if found_keywords:
                    features[f"has_{category}"] = True
                    features[f"{category}_keywords"] = found_keywords
        else:
            # Kiểm tra các loại keywords exact matching
            for category, keywords in self.knowledge_base["context_keywords"].items():
                found_keywords = [kw for kw in keywords if kw in text_lower]
                if found_keywords:
                    features[f"has_{category}"] = True
                    features[f"{category}_keywords"] = found_keywords
        
        # Extract entities sử dụng entity extractor
        entities = self.entity_extractor.extract_entities(text)
        
        # Update features với entities
        for entity_type, values in entities.items():
            if values:
                features[f"has_{entity_type}"] = True
                features[f"{entity_type}_entities"] = values
        
        # Thêm thông tin context từ conversation history
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
        
        # Áp dụng basic context rules
        for rule_category, rules in self.context_rules.items():
            if rule_category == "multi_turn_context" and "previous_intent" not in context_features:
                continue
                
            for rule in rules:
                # Xử lý multi-turn context rules
                if rule_category == "multi_turn_context":
                    if context_features.get("previous_intent") == rule.get("previous_intent"):
                        # Kiểm tra keywords
                        keywords = rule.get("keywords", [])
                        if any(kw in text.lower() for kw in keywords):
                            adjusted_intent = rule.get("intent", adjusted_intent)
                            adjusted_confidence += rule.get("confidence_boost", 0)
                            logger.debug(f"Applied multi-turn rule: {rule}")
                            break
                
                # Xử lý intent disambiguation rules
                elif rule_category == "intent_disambiguation":
                    ambiguous_intents = rule.get("ambiguous_intents", [])
                    if base_intent in ambiguous_intents:
                        # Kiểm tra keywords
                        keywords = rule.get("keywords", [])
                        if any(kw in text.lower() for kw in keywords):
                            intent_to_use = rule.get("intent")
                            if intent_to_use:
                                adjusted_intent = intent_to_use
                                adjusted_confidence += rule.get("confidence_boost", 0)
                                logger.debug(f"Applied disambiguation rule: {rule}")
                                break
                
                # Xử lý basic context rules
                else:
                    keywords = rule.get("keywords", [])
                    required_keywords = rule.get("required_keywords", [])
                    
                    # Kiểm tra required keywords trước
                    has_required = True
                    if required_keywords:
                        has_required = any(rk in text.lower() for rk in required_keywords)
                    
                    if has_required:
                        # Nếu bật fuzzy matching
                        if self.config.get("enable_fuzzy_matching", True):
                            fuzzy_matches = self.fuzzy_matcher.contains_fuzzy(text.lower(), keywords)
                            has_match = len(fuzzy_matches) > 0
                        else:
                            has_match = any(kw in text.lower() for kw in keywords)
                        
                        if has_match:
                            rule_intent = rule.get("intent")
                            confidence_boost = rule.get("confidence_boost", 0)
                            
                            # Nếu intent hiện tại khác với intent được đề xuất bởi context
                            if base_intent != rule_intent:
                                # Tính similarity giữa base_intent và context intent
                                base_synonyms = " ".join(self.knowledge_base["intent_synonyms"][base_intent])
                                rule_synonyms = " ".join(self.knowledge_base["intent_synonyms"][rule_intent])
                                
                                base_similarity = self.calculate_semantic_similarity(base_synonyms, rule_synonyms)
                                
                                # Nếu similarity thấp, có thể context rule đúng hơn
                                if base_similarity < 0.5:
                                    adjusted_intent = rule_intent
                                    adjusted_confidence = base_confidence + confidence_boost
                                    logger.debug(f"Changed intent based on context rule: {base_intent} -> {rule_intent}")
                            else:
                                # Nếu intent giống nhau, tăng confidence
                                adjusted_confidence += confidence_boost
                                logger.debug(f"Boosted confidence for {base_intent} by {confidence_boost}")
        
        # Áp dụng entity-specific logic
        # Ví dụ: Nếu có time entity và intent là set-alarm hoặc set-reminder thì tăng confidence
        if context_features.get("has_time") and adjusted_intent in ["set-alarm", "set-reminder"]:
            adjusted_confidence += 0.1
            logger.debug(f"Boosted confidence for {adjusted_intent} due to time entity")
        
        # Nếu có person entity và intent là call hoặc send-mess thì tăng confidence
        if context_features.get("has_person") and adjusted_intent in ["call", "send-mess"]:
            adjusted_confidence += 0.1
            logger.debug(f"Boosted confidence for {adjusted_intent} due to person entity")
        
        # Cap confidence ở mức 1.0
        adjusted_confidence = min(adjusted_confidence, 1.0)
        
        return adjusted_intent, adjusted_confidence
    
    def pattern_matching(self, text: str) -> List[Tuple[str, float]]:
        """Pattern matching để nhận diện intent"""
        text_lower = text.lower()
        pattern_scores = []
        
        for pattern_type, patterns in self.semantic_patterns.items():
            intent = pattern_type.split("_")[0]  # Lấy tên intent từ tên pattern
            
            if pattern_type.endswith("_patterns"):
                intent = pattern_type[:-9]  # Remove "_patterns" suffix
            
            # Map một số pattern đặc biệt sang intent
            intent_mapping = {
                "alarm": "set-alarm",
                "message": "send-mess",
                "weather": "check-weather",
                "media": "play-media",
                "news": "read-news",
                "health": "check-health-status",
                "conversation": "general-conversation"
            }
            
            if intent in intent_mapping:
                intent = intent_mapping[intent]
            
            max_score = 0
            for pattern in patterns:
                # Nếu bật fuzzy matching, thử cả pattern matching lẫn fuzzy matching
                if self.config.get("enable_fuzzy_matching", True):
                    # Regular pattern matching
                    matches = re.findall(pattern, text_lower)
                    if matches:
                        # Tính score dựa trên độ dài match và vị trí
                        for match in matches:
                            match_text = match[0] if isinstance(match, tuple) else match
                            score = len(match_text) / len(text_lower) * 0.5
                            if match_text in text_lower[:len(text_lower)//2]:  # Match ở đầu câu
                                score += 0.3
                            max_score = max(max_score, score)
                    
                    # Fuzzy pattern matching cho các pattern đơn giản
                    # Chuyển pattern thành plain text (nếu có thể)
                    try:
                        plain_pattern = pattern.replace(r".*", " ").replace(r"(?:", "").replace(")", "")
                        plain_pattern = re.sub(r"[\(\)\[\]\{\}\?\*\+\|\\]", "", plain_pattern).strip()
                        
                        if plain_pattern and len(plain_pattern) > 3:
                            # Chỉ dùng fuzzy matching cho pattern đủ dài và có ý nghĩa
                            fuzzy_ratio = fuzz.token_set_ratio(plain_pattern, text_lower)
                            fuzzy_score = fuzzy_ratio / 100.0 * 0.4  # Scale và giảm weight so với exact match
                            max_score = max(max_score, fuzzy_score)
                    except:
                        pass  # Skip nếu không thể xử lý pattern
                else:
                    # Chỉ dùng regular pattern matching
                    matches = re.findall(pattern, text_lower)
                    if matches:
                        # Tính score dựa trên độ dài match và vị trí
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
        
        # Check multi-intent indicators first
        for multi_intent, indicators in self.knowledge_base.get("multi_intent_indicators", {}).items():
            score = 0
            matched_indicators = []
            
            for indicator in indicators:
                if self.config.get("enable_fuzzy_matching", True):
                    # Fuzzy matching
                    fuzzy_matches = self.fuzzy_matcher.contains_fuzzy(text_lower, [indicator])
                    if fuzzy_matches:
                        keyword, match_score = fuzzy_matches[0]
                        score += 0.2 * (match_score / 100)
                        matched_indicators.append(keyword)
                else:
                    # Exact matching
                    if indicator in text_lower:
                        score += 0.2
                        matched_indicators.append(indicator)
            
            if score > 0:
                # Check which intent is more likely based on the text and matched indicators
                intents = multi_intent.split(",")
                best_intent = intents[0]  # Default to first
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
        
        # Check single intent indicators
        for intent, indicators in self.knowledge_base["intent_indicators"].items():
            score = 0
            matched_indicators = []
            
            if self.config.get("enable_fuzzy_matching", True):
                # Fuzzy matching
                fuzzy_matches = self.fuzzy_matcher.contains_fuzzy(text_lower, indicators)
                for keyword, match_score in fuzzy_matches:
                    score += 0.2 * (match_score / 100)
                    matched_indicators.append(keyword)
                    
                    # Bonus cho exact match
                    if f" {keyword} " in f" {text_lower} ":
                        score += 0.1
            else:
                # Exact matching
                for indicator in indicators:
                    if indicator in text_lower:
                        score += 0.2
                        matched_indicators.append(indicator)
                        
                        # Bonus cho exact match
                        if f" {indicator} " in f" {text_lower} ":
                            score += 0.1
            
            if score > 0:
                # Check position - indicators at the beginning get a boost
                for indicator in matched_indicators:
                    if text_lower.startswith(indicator) or text_lower.find(f" {indicator}") < len(text_lower) // 3:
                        score += 0.1
                        break
                
                keyword_scores.append((intent, min(score, 1.0)))
        
        return keyword_scores
    
    def reasoning_predict(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Predict intent sử dụng reasoning engine"""
        start_time = time.time()
        logger.info(f"🧠 REASONING ENGINE: Phân tích text: '{text}'")
        
        # Kiểm tra trong cache
        cached_result = self.cache.get_result(text)
        if cached_result is not None and self.config.get("enable_cache", True):
            logger.info(f"Đã tìm thấy kết quả trong cache cho: '{text}'")
            return cached_result
        
        # Sử dụng context nếu được cung cấp
        if context:
            self.conversation_context.session_data.update(context)
        
        # 1. Semantic similarity với knowledge base
        semantic_results = self.find_similar_intents(text)
        logger.info(f"📊 Semantic similarity results: {semantic_results}")
        
        # 2. Pattern matching
        pattern_results = self.pattern_matching(text)
        logger.info(f"🔍 Pattern matching results: {pattern_results}")
        
        # 3. Keyword matching
        keyword_results = self.keyword_matching(text)
        logger.info(f"🔑 Keyword matching results: {keyword_results}")
        
        # 4. Extract entities
        entities = self.entity_extractor.extract_entities(text)
        logger.info(f"👤 Extracted entities: {entities}")
        
        # 5. Extract context features
        context_features = self.extract_context_features(text)
        logger.info(f"🌐 Context features: {context_features}")
        
        # 6. Kết hợp các kết quả
        combined_scores = defaultdict(float)
        
        # Cộng dồn scores từ các phương pháp với weights từ config
        # Điều chỉnh weights để ưu tiên pattern matching và keyword matching hơn
        semantic_weight = self.config.get("semantic_weight", 0.25)  # Giảm từ 0.4 xuống 0.25
        pattern_weight = self.config.get("pattern_weight", 0.45)    # Tăng từ 0.35 lên 0.45
        keyword_weight = self.config.get("keyword_weight", 0.30)    # Tăng từ 0.25 lên 0.30
        
        for intent, score in semantic_results:
            combined_scores[intent] += score * semantic_weight
        
        for intent, score in pattern_results:
            combined_scores[intent] += score * pattern_weight
        
        for intent, score in keyword_results:
            combined_scores[intent] += score * keyword_weight
        
        # 7. Áp dụng context rules
        if combined_scores:
            best_intent = max(combined_scores.items(), key=lambda x: x[1])
            base_intent, base_confidence = best_intent
            
            # Áp dụng context rules
            adjusted_intent, adjusted_confidence = self.apply_context_rules(
                text, base_intent, base_confidence, context_features
            )
            
            logger.info(f"🎯 Context adjustment: {base_intent} ({base_confidence:.3f}) -> {adjusted_intent} ({adjusted_confidence:.3f})")
            
            # 8. Validation và confidence adjustment
            validation_result = self.validate_reasoning_result(
                text, adjusted_intent, adjusted_confidence, entities
            )
            
            # 9. Apply fallback strategy if needed
            fallback_result = self.apply_fallback_strategy(
                text, validation_result, semantic_results
            )
            
            # Build final result
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
                "explanation": self.generate_explanation(text, fallback_result["intent"], validation_result, entities),
                "suggestions": fallback_result.get("suggestions", []),
                "processing_time": time.time() - start_time
            }
            
            # Lưu vào cache
            if self.config.get("enable_cache", True):
                self.cache.set_result(text, result)
            
            # Cập nhật conversation context
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
            
            # Lưu vào cache
            if self.config.get("enable_cache", True):
                self.cache.set_result(text, unknown_result)
            
            # Cập nhật conversation context
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
        
        # Kiểm tra confidence threshold
        if confidence < self.similarity_threshold:
            validation["is_valid"] = False
            validation["warnings"].append(f"Confidence quá thấp: {confidence:.3f}")
            validation["suggestions"].append("Cần xác nhận lại từ người dùng")
        
        # Kiểm tra mâu thuẫn
        text_lower = text.lower()
        conflicting_patterns = {
            "call": ["nhắn tin", "gửi tin nhắn", "soạn tin"],
            "send-mess": ["gọi", "điện thoại", "alo"],
            "set-alarm": ["gửi tin nhắn", "gọi", "nhắn tin"],
            "play-media": ["gọi", "nhắn tin", "kiểm tra"]
        }
        
        if intent in conflicting_patterns:
            conflicting = conflicting_patterns[intent]
            # Sử dụng fuzzy matching nếu được bật
            if self.config.get("enable_fuzzy_matching", True):
                fuzzy_matches = self.fuzzy_matcher.contains_fuzzy(text_lower, conflicting)
                if fuzzy_matches:
                    validation["confidence"] *= 0.7
                    validation["warnings"].append("Có từ khóa mâu thuẫn với intent")
            else:
                if any(pattern in text_lower for pattern in conflicting):
                    validation["confidence"] *= 0.7
                    validation["warnings"].append("Có từ khóa mâu thuẫn với intent")
        
        # Kiểm tra entity requirements cho các intent
        intent_entity_requirements = {
            "call": ["person"],
            "send-mess": ["person"],
            "set-alarm": ["time"],
            "set-reminder": ["time", "action"],
            "check-weather": ["location", "time"],
            "play-media": ["object"]
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
                
                # Thêm gợi ý cho người dùng
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
        
        # Kiểm tra context completeness từ conversation history
        current_context = self.conversation_context.get_current_context()
        if current_context["current_intent"] and current_context["current_intent"] != intent:
            # Nếu intent hiện tại khác với intent trước đó, có thể là intent mới hoặc cần disambiguate
            # Kiểm tra xem có entities nào từ lượt trước có thể áp dụng không
            if intent in intent_entity_requirements and entities:
                required_entities = intent_entity_requirements[intent]
                for entity_type in required_entities:
                    if (entity_type not in entities or not entities[entity_type]) and \
                       entity_type in current_context["current_entities"] and \
                       current_context["current_entities"][entity_type]:
                        # Nếu entity được yêu cầu không có trong lượt hiện tại nhưng có trong context
                        # Tăng confidence vì có thể user đang tiếp tục hội thoại
                        validation["confidence"] *= 1.1
                        validation["warnings"].append(f"Sử dụng {entity_type} từ context trước đó")
        
        return validation
    
    def apply_fallback_strategy(self, text: str, validation: Dict[str, Any], 
                              semantic_results: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Áp dụng fallback strategy dựa trên confidence và validation"""
        intent = validation["intent"]
        confidence = validation["confidence"]
        
        fallback_result = {
            "intent": intent,
            "confidence": confidence
        }
        
        # Lấy thresholds từ config
        thresholds = self.intent_fallback["confidence_thresholds"]
        
        # Nếu confidence quá thấp, fallback sang unknown hoặc help
        if confidence < thresholds["very_low"]:
            fallback_result["intent"] = self.intent_fallback["fallback_intents"]["default"]
            fallback_result["confidence"] = 0.2
            fallback_result["fallback_info"] = {
                "original_intent": intent,
                "original_confidence": confidence,
                "reason": "confidence_too_low"
            }
            fallback_result["suggestions"] = self.intent_fallback["intent_suggestions"]["unknown"]
        
        # Nếu confidence thấp, nhưng có cảnh báo, fallback sang clarification
        elif confidence < thresholds["low"] and validation.get("warnings"):
            fallback_result["intent"] = self.intent_fallback["fallback_intents"]["clarification"]
            fallback_result["confidence"] = 0.4
            fallback_result["fallback_info"] = {
                "original_intent": intent,
                "original_confidence": confidence,
                "reason": "low_confidence_with_warnings",
                "warnings": validation.get("warnings", [])
            }
            
            # Thêm gợi ý cho người dùng
            if intent in self.intent_fallback["intent_suggestions"]:
                fallback_result["suggestions"] = self.intent_fallback["intent_suggestions"][intent]
            else:
                fallback_result["suggestions"] = self.intent_fallback["intent_suggestions"]["unknown"]
        
        # Nếu confidence medium, nhưng có ít nhất 2 intent có confidence gần nhau
        elif confidence < thresholds["medium"] and len(semantic_results) >= 2:
            top_intents = semantic_results[:2]
            if abs(top_intents[0][1] - top_intents[1][1]) < 0.1:
                # Ambiguous intents
                fallback_result["intent"] = self.intent_fallback["fallback_intents"]["clarification"]
                fallback_result["confidence"] = 0.5
                fallback_result["fallback_info"] = {
                    "original_intent": intent,
                    "original_confidence": confidence,
                    "reason": "ambiguous_intents",
                    "ambiguous_intents": [i[0] for i in top_intents]
                }
                
                # Combine suggestions from both intents
                suggestions = []
                for ambiguous_intent, _ in top_intents:
                    if ambiguous_intent in self.intent_fallback["intent_suggestions"]:
                        suggestions.extend(self.intent_fallback["intent_suggestions"][ambiguous_intent][:1])
                
                if not suggestions:
                    suggestions = self.intent_fallback["intent_suggestions"]["unknown"]
                
                fallback_result["suggestions"] = suggestions
        
        # Nếu confidence ok nhưng có nhiều cảnh báo, thêm suggestions
        elif validation.get("warnings") and len(validation.get("warnings", [])) > 1:
            # Keep the original intent but add suggestions
            fallback_result["suggestions"] = validation.get("suggestions", [])
            
            # Nếu không có suggestions từ validation, lấy từ intent_suggestions
            if not fallback_result["suggestions"] and intent in self.intent_fallback["intent_suggestions"]:
                fallback_result["suggestions"] = self.intent_fallback["intent_suggestions"][intent]
        
        # Nếu confidence cao, không cần fallback
        else:
            # Vẫn thêm suggestions nếu có
            if validation.get("suggestions"):
                fallback_result["suggestions"] = validation.get("suggestions")
            elif intent in self.intent_fallback["intent_suggestions"]:
                # Nếu confidence khá cao, chỉ cần 1 gợi ý
                fallback_result["suggestions"] = [self.intent_fallback["intent_suggestions"][intent][0]] \
                                               if self.intent_fallback["intent_suggestions"][intent] else []
        
        return fallback_result
    
    def generate_explanation(self, text: str, intent: str, validation: Dict, 
                           entities: Optional[Dict[str, List[str]]] = None) -> str:
        """Tạo explanation cho kết quả reasoning"""
        explanation_parts = []
        
        # Giải thích intent được chọn
        explanation_parts.append(f"Intent '{intent}' được chọn với confidence {validation['confidence']:.3f}")
        
        # Giải thích entities
        if entities:
            entity_explanations = []
            for entity_type, values in entities.items():
                if values:
                    entity_explanations.append(f"{entity_type}: {', '.join(values)}")
            
            if entity_explanations:
                explanation_parts.append(f"Entities: {'; '.join(entity_explanations)}")
        
        # Giải thích các warnings
        if validation.get("warnings"):
            explanation_parts.append(f"Cảnh báo: {', '.join(validation['warnings'])}")
        
        # Giải thích context
        context_features = validation.get("context_features", {})
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
        
        # Kiểm tra conversation history
        current_context = self.conversation_context.get_current_context()
        if current_context.get("previous_intent"):
            explanation_parts.append(f"Previous intent: {current_context['previous_intent']}")
        
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
        
        # Cập nhật vector store
        if self.config.get("enable_vectorstore", True):
            self._initialize_vector_store()
    
    def save_knowledge_base(self, file_path: str = None):
        """Lưu knowledge base"""
        if file_path is None:
            file_path = self.config.get("knowledge_base_path", "knowledge_base.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Đã lưu knowledge base vào {file_path}")
    
    def load_knowledge_base(self, file_path: str = None):
        """Load knowledge base"""
        if file_path is None:
            file_path = self.config.get("knowledge_base_path", "knowledge_base.json")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)
        
        # Cập nhật vector store
        if self.config.get("enable_vectorstore", True):
            self._initialize_vector_store()
        
        logger.info(f"Đã load knowledge base từ {file_path}")
    
    def reset_conversation_context(self):
        """Reset conversation context"""
        self.conversation_context.reset()
        logger.info("Đã reset conversation context")
    
    def clear_cache(self):
        """Clear cache"""
        self.cache.clear()
        logger.info("Đã clear cache")

# Test function
def test_reasoning_engine():
    """Test reasoning engine"""
    print("🧪 TESTING REASONING ENGINE")
    print("=" * 50)
    
    # Khởi tạo engine với config mặc định
    engine = ReasoningEngine()
    
    # Test cases với các từ ngữ không có trong dataset
    test_cases = [
        "kết nối với mẹ tôi qua điện thoại",
        "nhắc tôi uống thuốc lúc 8 giờ sáng mai",
        "dự báo thời tiết hôm nay thế nào",
        "bật nhạc bolero cho tôi nghe",
        "cập nhật tin tức mới nhất",
        "soạn tin nhắn gửi cho con trai",
        "đặt chuông báo thức 6 giờ sáng mai",
        "kiểm tra chỉ số huyết áp của tôi",
        "chào bạn, hôm nay bạn thế nào",
        "kết nối internet cho tôi",
        "tìm kiếm thông tin về bệnh tiểu đường",
        "đặt lịch hẹn với bác sĩ",
        # Test fuzzy matching
        "goi dien cho me toi",  # Thiếu dấu "gọi điện cho mẹ tôi"
        "dat bao thuc 6h sang",  # Thiếu dấu "đặt báo thức 6h sáng"
        "nhan tin cho ban toi",  # Thiếu dấu "nhắn tin cho bạn tôi"
        # Test multi-turn context
        "đặt báo thức",  # Turn 1
        "8 giờ sáng mai",  # Turn 2 - should understand this is related to previous alarm intent
        "gọi điện",  # Turn 1
        "cho mẹ tôi",  # Turn 2 - should understand this is related to previous call intent
        # Test ambiguous intents
        "nhắc tôi gọi điện cho bác sĩ vào ngày mai",  # Both reminder and call
        "gửi tin nhắn cho mẹ tôi nhắc bà uống thuốc"  # Both message and reminder
    ]
    
    # Test multi-turn conversations
    print("\n🔄 TESTING MULTI-TURN CONVERSATIONS")
    print("-" * 40)
    
    # Conversation 1
    print("\n📱 Conversation 1: Setting an alarm")
    engine.reset_conversation_context()
    
    turn1 = "đặt báo thức"
    print(f"\nUser: {turn1}")
    result1 = engine.reasoning_predict(turn1)
    print(f"🤖 Intent: {result1['intent']} (Confidence: {result1['confidence']:.3f})")
    if result1.get('suggestions'):
        print(f"💡 Suggestion: {result1['suggestions'][0]}")
    
    turn2 = "8 giờ sáng mai"
    print(f"\nUser: {turn2}")
    result2 = engine.reasoning_predict(turn2)
    print(f"🤖 Intent: {result2['intent']} (Confidence: {result2['confidence']:.3f})")
    print(f"🔍 Entities: {result2.get('entities', {})}")
    
    # Conversation 2
    print("\n📱 Conversation 2: Making a call")
    engine.reset_conversation_context()
    
    turn1 = "gọi điện"
    print(f"\nUser: {turn1}")
    result1 = engine.reasoning_predict(turn1)
    print(f"🤖 Intent: {result1['intent']} (Confidence: {result1['confidence']:.3f})")
    if result1.get('suggestions'):
        print(f"💡 Suggestion: {result1['suggestions'][0]}")
    
    turn2 = "cho mẹ tôi"
    print(f"\nUser: {turn2}")
    result2 = engine.reasoning_predict(turn2)
    print(f"🤖 Intent: {result2['intent']} (Confidence: {result2['confidence']:.3f})")
    print(f"🔍 Entities: {result2.get('entities', {})}")
    
    # Test individual cases
    print("\n🔍 TESTING INDIVIDUAL CASES")
    print("-" * 40)
    
    for i, text in enumerate(test_cases[:12], 1):  # Test first 12 cases
        print(f"\n📝 Test case {i}: '{text}'")
        print("-" * 40)
        
        result = engine.reasoning_predict(text)
        
        print(f"🎯 Intent: {result['intent']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        print(f"🔍 Method: {result['reasoning_method']}")
        if result.get('entities'):
            print(f"👤 Entities: {result['entities']}")
        print(f"💡 Explanation: {result['explanation']}")
        
        if result.get('suggestions'):
            print(f"💭 Suggestions: {result['suggestions']}")
        
        if result.get('validation') and result['validation'].get('warnings'):
            print(f"⚠️  Warnings: {result['validation']['warnings']}")
    
    # Test fuzzy matching
    print("\n🔤 TESTING FUZZY MATCHING")
    print("-" * 40)
    
    for i, text in enumerate(test_cases[12:15], 1):
        print(f"\n📝 Fuzzy test {i}: '{text}'")
        print("-" * 40)
        
        result = engine.reasoning_predict(text)
        
        print(f"🎯 Intent: {result['intent']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        if result.get('entities'):
            print(f"👤 Entities: {result['entities']}")
        print(f"💡 Explanation: {result['explanation']}")
        
        if result.get('suggestions'):
            print(f"💭 Suggestions: {result['suggestions']}")
    
    # Test ambiguous intents
    print("\n🤔 TESTING AMBIGUOUS INTENTS")
    print("-" * 40)
    
    for i, text in enumerate(test_cases[-2:], 1):
        print(f"\n📝 Ambiguous test {i}: '{text}'")
        print("-" * 40)
        
        result = engine.reasoning_predict(text)
        
        print(f"🎯 Intent: {result['intent']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        if result.get('entities'):
            print(f"👤 Entities: {result['entities']}")
        print(f"💡 Explanation: {result['explanation']}")
        
        if result.get('suggestions'):
            print(f"💭 Suggestions: {result['suggestions']}")
    
    # Test performance
    print("\n⏱️ TESTING PERFORMANCE")
    print("-" * 40)
    
    # Clear cache first
    engine.clear_cache()
    
    start_time = time.time()
    for _ in range(3):  # Run a few iterations to measure performance
        for text in test_cases[:5]:  # Use first 5 test cases
            _ = engine.reasoning_predict(text)
    
    total_time = time.time() - start_time
    avg_time = total_time / (3 * 5)
    print(f"Average processing time per request: {avg_time:.4f} seconds")
    
    # Test with cache
    start_time = time.time()
    for _ in range(3):  # Run with cache
        for text in test_cases[:5]:  # Use first 5 test cases
            _ = engine.reasoning_predict(text)
    
    cache_time = time.time() - start_time
    print(f"Average processing time with cache: {cache_time/15:.4f} seconds")
    print(f"Cache speedup: {total_time/cache_time:.2f}x")
    
    print("\n🎉 REASONING ENGINE TEST COMPLETED!")

if __name__ == "__main__":
    test_reasoning_engine()