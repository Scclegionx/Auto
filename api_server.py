import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import re
from datetime import datetime
import json
import os
from config import model_config
from reasoning_engine import ReasoningEngine

# Pydantic models
class IntentRequest(BaseModel):
    text: str
    confidence_threshold: Optional[float] = 0.3

class IntentResponse(BaseModel):
    input_text: str
    intent: str
    confidence: float
    command: str
    entities: Dict[str, str]
    value: str
    processing_time: float
    timestamp: str

class SimpleIntentModel(nn.Module):
    """Model tối ưu cho Intent Recognition với Large model và GPU"""
    
    def __init__(self, model_name, num_intents, config):
        super().__init__()
        self.config = config
        
        # Load PhoBERT model với gradient checkpointing để tiết kiệm memory
        self.phobert = AutoModel.from_pretrained(
            model_name,
            gradient_checkpointing=config.gradient_checkpointing
        )
        
        # Multi-layer classifier cho large model
        hidden_size = self.phobert.config.hidden_size
        
        if config.model_size == "large":
            # Large model: sử dụng multi-layer classifier
            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size // 2),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size // 4),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_size // 4, num_intents)
            )
        else:
            # Base model: sử dụng simple classifier
            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(hidden_size, num_intents)
            )
    
    def forward(self, input_ids, attention_mask):
        # Sử dụng gradient checkpointing nếu được bật
        if self.config.gradient_checkpointing and self.training:
            outputs = self.phobert(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                use_cache=False  # Tắt cache để tiết kiệm memory
            )
        else:
            outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Sử dụng mean pooling thay vì pooler_output cho ổn định hơn
        sequence_output = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled_output = (sequence_output * attention_mask_expanded).sum(dim=1) / attention_mask_expanded.sum(dim=1)
        
        logits = self.classifier(pooled_output)
        return logits

class PhoBERT_SAM_API:
    """API class cho PhoBERT_SAM"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.id_to_intent = None
        self.intent_to_command = None
        
        # Auto-detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {self.device}")
        
        if self.device.type == "cuda":
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Intent to Command mapping - Cap nhat theo 28 commands tu dataset
        self.intent_to_command = {
            "adjust-settings": "adjust_settings",
            "app-tutorial": "app_tutorial", 
            "browse-social-media": "browse_social_media",
            "call": "call",
            "check-device-status": "check_device_status",
            "check-health-status": "check_health_status",
            "check-messages": "check_messages",
            "check-weather": "check_weather",
            "control-device": "control_device",
            "general-conversation": "general_conversation",
            "help": "help",
            "make-call": "make_call",
            "make-video-call": "make_video_call",
            "navigation-help": "navigation_help",
            "open-app": "open_app",
            "open-app-action": "open_app_action",
            "play-audio": "play_audio",
            "play-content": "play_content",
            "play-media": "play_media",
            "provide-instructions": "provide_instructions",
            "read-content": "read_content",
            "read-news": "read_news",
            "search-content": "search_content",
            "search-internet": "search_internet",
            "send-message": "send_message",
            "send-mess": "send_mess",
            "set-alarm": "set_alarm",
            "set-reminder": "set_reminder",
            "view-content": "view_content",
            "unknown": "unknown"
        }
        
        # Entity patterns - Cải thiện để extract chính xác hơn
        self.entity_patterns = {
            "RECEIVER": [
                # Improved patterns for receiver extraction
                r"cho\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá|ngay bây giờ|ngay|bây giờ))?(?:$|[\.,])",  # "cho bà ngoại tôi ngay bây giờ"
                r"gọi\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá|ngay bây giờ|ngay|bây giờ))?(?:$|[\.,])",  # "gọi cho bà ngoại tôi"
                r"nhắn\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá|ngay bây giờ|ngay|bây giờ))?(?:$|[\.,])",  # "nhắn cho bà ngoại tôi"
                r"gửi\s+(?:cho|tới|đến)\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá|ngay bây giờ|ngay|bây giờ))?(?:$|[\.,])",  # "gửi cho bác Lan nhé"
                r"(?:báo|thông báo|nói|nói với|thông tin)\s+(?:cho|tới|đến)\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá|ngay bây giờ|ngay|bây giờ))?(?:$|[\.,])",  # "thông báo cho chị Hương"
                r"(?:số|số điện thoại|liên lạc với|liên hệ với)\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá|ngay bây giờ|ngay|bây giờ))?(?:$|[\.,])",  # "liên lạc với anh Tuấn"
                r"(?:kết nối|liên lạc|liên hệ)\s+(?:với|tới|đến)\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá|ngay bây giờ|ngay|bây giờ))?(?:$|[\.,])",  # "kết nối với bà"
                r"(?:với|cùng)\s+((?:bác|chú|cô|anh|chị|em|ông|bà)\s+[\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá|ngay bây giờ|ngay|bây giờ))?(?:$|[\.,])",  # "với bác Phương"
                r"(?:cuộc gọi|gọi điện|gọi thoại)\s+(?:cho|tới|đến)\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá|ngay bây giờ|ngay|bây giờ))?(?:$|[\.,])",  # "cuộc gọi cho bà ngoại tôi"
                r"(?:thực hiện|thực hiện một)\s+(?:cuộc gọi|gọi điện|gọi thoại)\s+(?:cho|tới|đến)\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá|ngay bây giờ|ngay|bây giờ))?(?:$|[\.,])"  # "thực hiện cuộc gọi cho bà ngoại tôi"
            ],
            
            "PLATFORM": [
                # Current patterns
                r"trên\s+(Facebook|Zalo|Youtube|YouTube|SMS|Messenger|Telegram|Instagram|TikTok)",  # "trên Zalo"
                r"qua\s+(Facebook|Zalo|Youtube|YouTube|SMS|Messenger|Telegram|Instagram|TikTok)",
                r"bằng\s+(Facebook|Zalo|Youtube|YouTube|SMS|Messenger|Telegram|Instagram|TikTok)",
                r"(Facebook|Zalo|Youtube|YouTube|SMS|Messenger|Telegram|Instagram|TikTok)\s+app",
                r"app\s+(Facebook|Zalo|Youtube|YouTube|SMS|Messenger|Telegram|Instagram|TikTok)",
                r"(Facebook|Zalo|Youtube|YouTube|SMS|Messenger|Telegram|Instagram|TikTok)",  # Trực tiếp
                r"vào\s+(Facebook|Zalo|Youtube|YouTube|SMS|Messenger|Telegram|Instagram|TikTok)",  # "vào Facebook"
                
                # Additional patterns
                r"(?:sử dụng|dùng|thông qua|qua đường|đường)\s+(Facebook|Zalo|Youtube|YouTube|SMS|Messenger|Telegram|Instagram|TikTok|tin nhắn|điện thoại)",  # "sử dụng Zalo"
                r"(?:ứng dụng|phần mềm)\s+(Facebook|Zalo|Youtube|YouTube|SMS|Messenger|Telegram|Instagram|TikTok)",  # "ứng dụng Facebook"
                r"(?:mở|vào|khởi động)\s+(?:ứng dụng|phần mềm)?\s*(Facebook|Zalo|Youtube|YouTube|SMS|Messenger|Telegram|Instagram|TikTok)",  # "mở ứng dụng Zalo"
                r"(?:nhắn tin|gửi tin nhắn|chat)\s+(?:qua|trên|bằng|dùng)?\s*(Facebook|Zalo|Youtube|YouTube|SMS|Messenger|Telegram|Instagram|TikTok)",  # "nhắn tin qua Zalo"
                r"(?:gọi|gọi điện|video call|cuộc gọi|facetime)\s+(?:qua|trên|bằng|dùng)?\s*(Facebook|Zalo|Youtube|YouTube|SMS|Messenger|Telegram|Instagram|TikTok)",  # "gọi điện qua Zalo"
                
                # Improved patterns for search and content
                r"(?:tìm kiếm|tìm|search|tra cứu)\s+(?:trên|qua|bằng|dùng)?\s*(Facebook|Zalo|Youtube|YouTube|SMS|Messenger|Telegram|Instagram|TikTok)",  # "tìm kiếm trên Youtube"
                r"(?:xem|phát|nghe|mở)\s+(?:trên|qua|bằng|dùng)?\s*(Facebook|Zalo|Youtube|YouTube|SMS|Messenger|Telegram|Instagram|TikTok)",  # "xem trên Youtube"
                r"(?:vào|mở)\s+(Facebook|Zalo|Youtube|YouTube|SMS|Messenger|Telegram|Instagram|TikTok)\s+(?:để|để mà|mà)\s+(?:tìm|tìm kiếm|search|phát|nghe|xem)"  # "vào Youtube để tìm kiếm"
            ],
            
            "TIME": [
                # Current patterns
                r"(\d{1,2}:\d{2})",
                r"(\d{1,2})\s*giờ",
                r"(\d{1,2})\s*phút",
                r"(sáng|chiều|tối|đêm)",
                r"(hôm\s+nay|ngày\s+mai|tuần\s+sau)",
                
                # Additional patterns
                r"(\d{1,2})\s*giờ\s*(\d{1,2})?\s*(?:phút)?",  # "7 giờ 30 phút", "7 giờ 30", "7 giờ"
                r"(\d{1,2})\s*rưỡi",  # "7 rưỡi"
                r"(\d{1,2})\s*giờ\s*rưỡi",  # "7 giờ rưỡi"
                r"(\d{1,2})\s*giờ\s*(?:kém|thiếu)\s*(\d{1,2})",  # "7 giờ kém 15"
                r"(\d{1,2})\s*giờ\s*(\d{1,2})\s*(?:phút)?\s*(?:sáng|trưa|chiều|tối|đêm)",  # "7 giờ 30 phút sáng"
                r"(\d{1,2})\s*(?:giờ)?\s*(?:sáng|trưa|chiều|tối|đêm)",  # "7 giờ sáng", "7 sáng"
                r"(?:lúc|vào\s+lúc|vào)\s+(\d{1,2})\s*(?:giờ|h|:)\s*(\d{1,2})?(?:\s*phút)?",  # "lúc 7 giờ", "vào lúc 7:30"
                r"(?:hôm\s+nay|ngày\s+mai|ngày\s+kia|tuần\s+sau|tuần\s+tới)",  # "hôm nay", "tuần tới"
                r"(?:thứ\s+[Hh]ai|thứ\s+[Bb]a|thứ\s+[Tt]ư|thứ\s+[Nn]ăm|thứ\s+[Ss]áu|thứ\s+[Bb]ảy|chủ\s+nhật)",  # "thứ hai", "chủ nhật"
                r"(?:sáng|trưa|chiều|tối|đêm)\s+(?:nay|mai|kia)",  # "sáng nay", "tối mai"
                r"(?:ngày|mùng|mồng)\s+(\d{1,2})(?:\s+tháng\s+(\d{1,2}))?(?:\s+năm\s+(\d{4}))?",  # "ngày 15", "ngày 15 tháng 8"
                r"(\d{1,2})\/(\d{1,2})(?:\/(\d{4}))?",  # "15/8", "15/8/2023"
                r"(?:vài|mấy|mười|hai\s+mươi|ba\s+mươi)\s+(?:giây|phút|tiếng|ngày|tuần|tháng)\s+(?:tới|sau|nữa)",  # "vài phút nữa", "mười ngày tới"
                r"(?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\s+(?:giây|phút|tiếng|ngày|tuần|tháng)\s+(?:tới|sau|nữa)"  # "hai tiếng nữa"
            ],
            
            "MESSAGE": [
                # Improved patterns for message extraction
                r"nói\s+rõ\s+là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "nói rõ là tôi muốn trò chuyện với bà"
                r"rằng\s+là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "rằng là chiều nay đón bà"
                r"rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "rằng chiều nay 6 giờ chiều đón bà"
                r"nói\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "nói tôi muốn trò chuyện"
                r"nhắn\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "nhắn tôi sẽ đến"
                r"gửi\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "gửi tôi sẽ đến"
                
                # Additional patterns
                r"(?:rằng|là)\s+[\"\'](.+?)[\"\']",  # Trích dẫn nội dung tin nhắn bằng dấu ngoặc kép hoặc đơn
                r"(?:nhắn|nhắn tin|gửi|gửi tin nhắn|nhắn lại|gửi lời nhắn)\s+(?:rằng|là)?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                r"(?:với\s+nội\s+dung|với\s+tin\s+nhắn|tin\s+nhắn)\s+(?:là|rằng)?\s+[\"\']?(.+?)[\"\']?(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                r"(?:nội\s+dung|tin\s+nhắn)\s*[\"\'](.+?)[\"\']",
                r"(?:nhắn\s+cho\s+\w+\s+)(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # Tin nhắn sau "nhắn cho [người nhận]"
                r"(?:gửi\s+cho\s+\w+\s+)(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])"  # Tin nhắn sau "gửi cho [người nhận]"
            ],
            
            "LOCATION": [
                # Current patterns
                r"ở\s+(\w+(?:\s+\w+)*)",
                r"tại\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+(?:thành\s+phố|tỉnh|quận|huyện)",
                
                # Additional patterns
                r"(?:ở|tại|trong|ngoài|gần|xa)\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "ở Hà Nội", "tại quận 1"
                r"(?:đến|tới|về|qua|sang|đi)\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "đến Sài Gòn", "về quê"
                r"(?:thành phố|tỉnh|quận|huyện|phường|xã|làng|thôn|ấp|khu|vùng)\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "thành phố Hồ Chí Minh"
                r"(?:trong|ngoài|gần|xa)\s+(?:thành phố|tỉnh|quận|huyện|phường|xã|làng|thôn|ấp|khu|vùng)\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "trong thành phố Đà Nẵng"
                r"(?:khu\s+vực|khu\s+đô\s+thị|khu\s+dân\s+cư|làng|xóm)\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "khu vực Mỹ Đình"
                r"(?:đường|phố|ngõ|ngách|hẻm)\s+([\w\s\d\/]+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "đường Lê Lợi", "ngõ 193"
                r"(?:số\s+nhà|nhà\s+số)\s+([\w\s\d\/]+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "số nhà 45", "nhà số 15"
                r"(?:toà\s+nhà|chung\s+cư|khu\s+chung\s+cư|căn\s+hộ)\s+([\w\s\d\/]+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "toà nhà CT1", "chung cư Linh Đàm"
                r"(?:bệnh\s+viện|trường\s+học|trường|trường\s+đại\s+học|đại\s+học|trường\s+phổ\s+thông|siêu\s+thị|chợ|cửa\s+hàng|công\s+ty)\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])"  # "bệnh viện Bạch Mai", "trường đại học Quốc Gia"
            ],
            
            "APP": [
                # Current patterns
                r"mở\s+(Facebook|Zalo|Youtube|YouTube|SMS|Messenger|Telegram|Instagram|TikTok|Camera|Gallery|Settings|Clock|Weather|Maps|Calculator)",
                r"vào\s+(Facebook|Zalo|Youtube|YouTube|SMS|Messenger|Telegram|Instagram|TikTok|Camera|Gallery|Settings|Clock|Weather|Maps|Calculator)",
                r"(Facebook|Zalo|Youtube|YouTube|SMS|Messenger|Telegram|Instagram|TikTok|Camera|Gallery|Settings|Clock|Weather|Maps|Calculator)\s+app",
                
                # Additional patterns
                r"(?:mở|khởi động|chạy|vào|sử dụng|dùng)\s+(?:ứng dụng|app|phần mềm)?\s*(Facebook|Zalo|Youtube|YouTube|TikTok|Instagram|Twitter|Messenger|Viber|Telegram|Google|Gmail|Chrome|Safari|Firefox|Opera|Maps|Bản đồ|Tin nhắn|SMS|Điện thoại|Máy tính|Calculator|Ghi âm|Ghi chú|Notes|Lịch|Calendar|Đồng hồ|Clock|Báo thức|Alarm|Thời tiết|Weather|Camera|Máy ảnh|Gallery|Bộ sưu tập|Hình ảnh|Cài đặt|Settings|Music|Nhạc|Video|Trò chơi|Game)",
                r"(?:ứng dụng|app|phần mềm)\s+(Facebook|Zalo|Youtube|YouTube|TikTok|Instagram|Twitter|Messenger|Viber|Telegram|Google|Gmail|Chrome|Safari|Firefox|Opera|Maps|Bản đồ|Tin nhắn|SMS|Điện thoại|Máy tính|Calculator|Ghi âm|Ghi chú|Notes|Lịch|Calendar|Đồng hồ|Clock|Báo thức|Alarm|Thời tiết|Weather|Camera|Máy ảnh|Gallery|Bộ sưu tập|Hình ảnh|Cài đặt|Settings|Music|Nhạc|Video|Trò chơi|Game)",
                r"(?:vào|truy cập|sử dụng)\s+(Facebook|Zalo|Youtube|YouTube|TikTok|Instagram|Twitter|Messenger|Viber|Telegram|Google|Gmail|Chrome|Safari|Firefox|Opera|Maps|Bản đồ|Tin nhắn|SMS|Điện thoại|Máy tính|Calculator|Ghi âm|Ghi chú|Notes|Lịch|Calendar|Đồng hồ|Clock|Báo thức|Alarm|Thời tiết|Weather|Camera|Máy ảnh|Gallery|Bộ sưu tập|Hình ảnh|Cài đặt|Settings|Music|Nhạc|Video|Trò chơi|Game)",
                r"(?:kiểm tra|xem|theo dõi)\s+(Facebook|Zalo|Youtube|YouTube|TikTok|Instagram|Twitter|Messenger|Viber|Telegram|Google|Gmail|Chrome|Safari|Firefox|Opera|Maps|Bản đồ|Tin nhắn|SMS|Điện thoại|Máy tính|Calculator|Ghi âm|Ghi chú|Notes|Lịch|Calendar|Đồng hồ|Clock|Báo thức|Alarm|Thời tiết|Weather|Camera|Máy ảnh|Gallery|Bộ sưu tập|Hình ảnh|Cài đặt|Settings|Music|Nhạc|Video|Trò chơi|Game)",
                r"(?:chụp ảnh|quay phim|quay video|xem ảnh|xem video)",  # Common app actions that imply app usage
                r"(?:tính toán|tính|làm tính|tính nhẩm)",  # Calculator
                r"(?:nghe nhạc|phát nhạc|bật nhạc)",  # Music app
                r"(?:ghi chú|note|ghi lại|lưu ý)",  # Notes app
                r"(?:đặt báo thức|hẹn giờ|đặt giờ)",  # Clock/Alarm app
                r"(?:thời tiết|dự báo|nhiệt độ)",  # Weather app
                r"(?:tìm đường|chỉ đường|định vị)"  # Maps app
            ],
            
            "QUERY": [
                # Current patterns
                r"tìm\s+(.+)",  # "tìm kiếm những thước phim hài"
                r"tìm\s+kiếm\s+(.+)",
                r"search\s+(.+)",
                r"tìm\s+video\s+(.+)",
                r"tìm\s+nhạc\s+(.+)",
                
                # Additional patterns
                r"(?:tìm|tìm kiếm|search|tra cứu|tra|tra cứu|kiếm|tìm hiểu)\s+(?:về|thông tin về|thông tin|kiến thức về|kiến thức)?\s*(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "tìm kiếm về cách nấu phở"
                r"(?:tìm|tìm kiếm|search|tra cứu|tra|tra cứu|kiếm|tìm hiểu)\s+(?:cho tôi|cho mình|cho tớ|cho bác|cho cô|cho chú|giúp tôi|giúp mình|giúp bác|giúp cô|giúp chú)?\s*(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "tìm cho tôi thông tin về sức khỏe"
                r"(?:tìm|tìm kiếm|search|tra cứu|tra|tra cứu|kiếm|tìm hiểu)\s+(?:video|clip|phim|nhạc|bài hát|bài|album|ca sĩ|ca khúc|nghệ sĩ|diễn viên|tác giả)\s+(?:về|của|do|bởi)?\s*(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "tìm video về cách làm bánh", "tìm nhạc của Trịnh Công Sơn"
                r"(?:tìm|tìm kiếm|search|tra cứu|tra|tra cứu|kiếm|tìm hiểu)\s+(?:công thức|cách|phương pháp|hướng dẫn|chỉ dẫn|bí quyết)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "tìm công thức nấu ăn", "tìm cách làm bánh"
                r"(?:tìm|tìm kiếm|search|tra cứu|tra|tra cứu|kiếm|tìm hiểu)\s+(?:tin tức|thời sự|báo|bản tin|thông tin)\s+(?:về|liên quan đến)?\s*(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "tìm tin tức về COVID-19"
                r"(?:tìm|tìm kiếm|search|tra cứu|tra|tra cứu|kiếm|tìm hiểu)\s+(?:bệnh|triệu chứng|thuốc|điều trị|bác sĩ|y tế|sức khỏe)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "tìm triệu chứng bệnh tiểu đường"
                r"(?:hỏi|tra cứu|tra|hỏi về|hỏi thông tin về)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "hỏi về cách sử dụng điện thoại"
                r"(?:tìm hiểu|nghiên cứu|học hỏi|học về)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "tìm hiểu về lịch sử Việt Nam"
                r"(?:cách|phương pháp|làm thế nào|làm sao|làm cách nào|làm như thế nào để)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "cách nấu phở", "làm thế nào để học tiếng Anh"
                
                # Improved patterns for complex search queries
                r"(?:vào|mở)\s+(?:youtube|facebook|zalo|instagram|tiktok)\s+(?:để|để mà|mà)\s+(?:tìm|tìm kiếm|search|phát|nghe|xem)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "vào youtube để tìm kiếm danh sách nhạc"
                r"(?:tìm kiếm|tìm|search)\s+(?:trên|qua|bằng|dùng)\s+(?:youtube|facebook|zalo|instagram|tiktok)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "tìm kiếm trên youtube danh sách nhạc"
                r"(?:danh sách|list|playlist)\s+(?:nhạc|music|video|clip|phim)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "danh sách nhạc mới nhất"
                r"(?:nhạc|music|video|clip|phim)\s+(?:mới nhất|hot|trending|phổ biến)\s+(?:của|do|bởi)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])"  # "nhạc mới nhất của Sơn Tùng"
            ],
            
            "CONTENT": [
                # Current patterns
                r"phim\s+(.+)",  # "phim hài của Xuân Bắc"
                r"nhạc\s+(.+)",
                r"video\s+(.+)",
                r"bài\s+hát\s+(.+)",
                r"tin\s+tức\s+(.+)",
                
                # Additional patterns
                r"(?:phim|video|clip|nhạc|bài hát|bài|album|ca khúc)\s+(?:về|của|do|bởi)?\s*(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "phim của Thành Long", "nhạc của Trịnh Công Sơn"
                r"(?:tin tức|thời sự|báo|bản tin|thông tin)\s+(?:về|liên quan đến)?\s*(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "tin tức về COVID-19"
                r"(?:phát|bật|mở|nghe|xem)\s+(?:phim|video|clip|nhạc|bài hát|bài|album|ca khúc)\s+(?:về|của|do|bởi)?\s*(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "bật nhạc của Sơn Tùng", "phát phim hài"
                r"(?:đọc|đọc báo|đọc tin|đọc tin tức)\s+(?:về|liên quan đến)?\s*(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "đọc báo về thời sự"
                r"(?:ca sĩ|nghệ sĩ|diễn viên|nhạc sĩ|tác giả|đạo diễn)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "ca sĩ Mỹ Tâm"
                r"(?:thể loại|loại|kiểu|dạng)\s+(?:phim|video|clip|nhạc|bài hát|bài|album|ca khúc)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "thể loại nhạc trữ tình"
                r"(?:phim|video|clip|nhạc|bài hát|bài|album|ca khúc)\s+(?:thể loại|loại|kiểu|dạng)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",  # "phim thể loại hài"
                r"(?:karaoke|hát karaoke)\s+(?:bài)?\s*(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])"  # "karaoke bài Đêm Lao Xao"
            ]
        }
        
        # Khởi tạo reasoning engine
        self.reasoning_engine = ReasoningEngine()
        print("🧠 Reasoning Engine initialized")
    
    def load_model(self):
        """Load trained model với hỗ trợ large model"""
        try:
            print("🔄 Loading PhoBERT Large model...")
            
            # Load config
            from config import ModelConfig
            model_config = ModelConfig()
            
            # Tìm model file - ưu tiên best model mới nhất
            model_dir = f"models/phobert_{model_config.model_size}_intent_model"
            
            # Tìm best model mới nhất
            best_model_path = None
            if os.path.exists(model_dir):
                best_models = []
                for filename in os.listdir(model_dir):
                    if filename.endswith('.pth') and 'best' in filename:
                        file_path = f"{model_dir}/{filename}"
                        # Lấy thông tin file để tìm model mới nhất
                        file_time = os.path.getmtime(file_path)
                        best_models.append((file_path, file_time))
                
                if best_models:
                    # Sắp xếp theo thời gian và lấy model mới nhất
                    best_models.sort(key=lambda x: x[1], reverse=True)
                    best_model_path = best_models[0][0]
                    print(f"🎯 Found latest best model: {os.path.basename(best_model_path)}")
            
            # Nếu không có best model, tìm model thường
            if not best_model_path:
                model_path = f"{model_dir}/model.pth"
                if not os.path.exists(model_path):
                    # Fallback to old model
                    model_path = "models/best_simple_intent_model.pth"
                    if not os.path.exists(model_path):
                        print(f"❌ No model found, using reasoning engine only")
                        self.model = None
                        self.tokenizer = None
                        self.id_to_intent = None
                        return True
            else:
                model_path = best_model_path
            
            print(f"📂 Loading model from: {model_path}")
            
            # Load checkpoint với map_location để tránh lỗi device
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Kiểm tra xem có metadata không
            if 'intent_to_id' not in checkpoint or 'id_to_intent' not in checkpoint:
                print("⚠️ Model không có metadata, sử dụng reasoning engine only")
                self.model = None
                self.tokenizer = None
                self.id_to_intent = None
                return True
            
            # Tạo model với config
            self.model = SimpleIntentModel(model_config.model_name, len(checkpoint['intent_to_id']), model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model.to(self.device)
            
            # Enable mixed precision nếu có GPU và config cho phép
            if self.device.type == "cuda" and model_config.use_fp16:
                self.model = self.model.half()  # Convert to FP16
                print("🔧 Enabled FP16 for GPU inference")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
            
            # Load mappings
            self.id_to_intent = checkpoint['id_to_intent']
            
            # Load thông tin model
            model_info = {
                'model_size': checkpoint.get('model_size', 'unknown'),
                'total_parameters': checkpoint.get('total_parameters', 0),
                'trainable_parameters': checkpoint.get('trainable_parameters', 0),
                'epoch': checkpoint.get('epoch', 'unknown'),
                'is_best': checkpoint.get('is_best', False),
                'validation_accuracy': checkpoint.get('validation_accuracy', 'unknown')
            }
            
            print(f"✅ Model loaded successfully from {model_path}")
            print(f"📊 Model file size: {os.path.getsize(model_path) / 1024**2:.2f} MB")
            print(f"🎯 Number of intents: {len(self.id_to_intent)}")
            print(f"🔧 Model info: {model_info}")
            print(f"📋 Available intents: {list(self.id_to_intent.values())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Sử dụng reasoning engine only")
            self.model = None
            self.tokenizer = None
            self.id_to_intent = None
            return True
    
    def extract_entities(self, text: str) -> Dict[str, str]:
        """Extract entities from text with improved logic"""
        entities = {}
        text_lower = text.lower()
        
        # Priority 1: Extract RECEIVER first (most important for call/message)
        receiver_patterns = [
            r"cho\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá|ngay bây giờ|ngay|bây giờ))?(?:$|[\.,])",
            r"gọi\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá|ngay bây giờ|ngay|bây giờ))?(?:$|[\.,])",
            r"nhắn\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá|ngay bây giờ|ngay|bây giờ))?(?:$|[\.,])",
            r"(?:cuộc gọi|gọi điện|gọi thoại)\s+(?:cho|tới|đến)\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá|ngay bây giờ|ngay|bây giờ))?(?:$|[\.,])",
            r"(?:thực hiện|thực hiện một)\s+(?:cuộc gọi|gọi điện|gọi thoại)\s+(?:cho|tới|đến)\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá|ngay bây giờ|ngay|bây giờ))?(?:$|[\.,])"
        ]
        
        for pattern in receiver_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match and match.group(1):
                receiver = match.group(1).strip()
                # Validate receiver has relationship terms
                relationship_terms = ["bố", "mẹ", "ông", "bà", "anh", "chị", "em", "con", "cháu", "chú", "bác", "cô", "dì", "ngoại", "nội"]
                if any(term in receiver for term in relationship_terms):
                    entities["RECEIVER"] = receiver
                    break
        
        # Priority 2: Extract PLATFORM
        platform_patterns = [
            r"bằng\s+(zalo|facebook|messenger|telegram|instagram|tiktok)",
            r"qua\s+(zalo|facebook|messenger|telegram|instagram|tiktok)",
            r"trên\s+(zalo|facebook|messenger|telegram|instagram|tiktok)",
            r"(zalo|facebook|messenger|telegram|instagram|tiktok)"
        ]
        
        for pattern in platform_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match and match.group(1):
                entities["PLATFORM"] = match.group(1).lower()
                break
        
        # Priority 3: Extract MESSAGE (only if we have a receiver)
        if "RECEIVER" in entities:
            receiver = entities["RECEIVER"]
            message_patterns = [
                rf"nói\s+rõ\s+là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                rf"rằng\s+là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                rf"rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                rf"nói\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])"
            ]
            
            for pattern in message_patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match and match.group(1):
                    message = match.group(1).strip()
                    # Clean up message - remove redundant words
                    message = re.sub(r'^(?:là\s+|rằng\s+)', '', message)
                    if message and len(message) > 5 and len(message) < 200:  # Reasonable length
                        entities["MESSAGE"] = message
                        break
        
        # Priority 4: Extract TIME (cải thiện cho nhắc nhở thuốc)
        time_patterns = [
            # Thời gian cụ thể
            r"(\d{1,2})\s*giờ\s*(\d{1,2})?\s*(?:phút)?",
            r"(\d{1,2})\s*rưỡi",
            r"(\d{1,2})\s*(?:giờ|h)\s*(?:sáng|trưa|chiều|tối)",
            # Thời gian tương đối
            r"(sáng|chiều|tối|đêm)",
            r"(hôm\s+nay|ngày\s+mai|tuần\s+sau)",
            r"(sau\s+(?:khi\s+)?ăn|sau\s+bữa\s+(?:sáng|trưa|tối))",
            r"(trước\s+(?:khi\s+)?ăn|trước\s+bữa\s+(?:sáng|trưa|tối))",
            # Thời gian định kỳ
            r"(hàng\s+ngày|hàng\s+tuần|hàng\s+tháng)",
            r"(thứ\s+\d+\s+hàng\s+tuần)",
            r"(ngày\s+\d+\s+hàng\s+tháng)",
            # Thời gian điều kiện
            r"(khi\s+cần\s+thiết|khi\s+đau|khi\s+có\s+triệu\s+chứng)",
            # Thời gian phức tạp
            r"(\d{1,2})\s*giờ\s*(?:sáng|trưa|chiều|tối)\s+và\s+(\d{1,2})\s*giờ\s*(?:sáng|trưa|chiều|tối)",
            r"(\d{1,2})\s*lần\s+một\s+ngày:\s*(sáng|trưa|tối)(?:,\s*(sáng|trưa|tối))*(?:,\s*(sáng|trưa|tối))*"
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                if match.groups():
                    time_value = " ".join([g for g in match.groups() if g])
                    if time_value:
                        entities["TIME"] = time_value
                        break
                else:
                    entities["TIME"] = match.group(0)
                    break
        
        # Priority 5: Extract LOCATION (if any)
        location_patterns = [
            r"ở\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"tại\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])"
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match and match.group(1):
                location = match.group(1).strip()
                if location and len(location) > 2:
                    entities["LOCATION"] = location
                    break
        
        # Special case: Check for medicine reminder words (highest priority)
        medicine_words = ["uống thuốc", "thuốc", "viên thuốc", "kháng sinh", "tiểu đường", "huyết áp", "tim", "vitamin", "sắt", "cảm", "đau đầu"]
        reminder_words = ["nhắc", "nhắc nhở", "ghi nhớ", "reminder", "đừng quên", "nhớ", "đặt lời nhắc", "đặt nhắc nhở"]
        
        if any(word in text_lower for word in medicine_words) and any(word in text_lower for word in reminder_words):
            # This is definitely a medicine reminder
            if "INTENT" not in entities:
                entities["INTENT"] = "set-reminder"
                # Extract medicine action as MESSAGE
                for medicine_word in medicine_words:
                    if medicine_word in text_lower:
                        # Find the full medicine action
                        medicine_patterns = [
                            rf"uống\s+(\d+\s+)?(?:viên\s+)?{medicine_word}",
                            rf"{medicine_word}\s+(?:lúc|vào|sau|trước)",
                            rf"uống\s+{medicine_word}"
                        ]
                        for pattern in medicine_patterns:
                            match = re.search(pattern, text_lower, re.IGNORECASE)
                            if match:
                                if match.groups():
                                    entities["MESSAGE"] = match.group(0)
                                else:
                                    entities["MESSAGE"] = match.group(0)
                                break
                        break
        
        # Special case: Check for message-related words
        elif any(word in text_lower for word in ["nhắn tin", "gửi tin", "soạn tin", "text", "sms", "message", "gửi", "nhắn"]):
            # This is likely a message sending intent
            if "INTENT" not in entities:
                entities["INTENT"] = "send-mess"
        
        # Special case: If we have a TIME but no specific intent, check for alarm/reminder words
        elif "TIME" in entities:
            if any(word in text_lower for word in ["báo thức", "đánh thức", "alarm", "dậy", "thức dậy"]):
                # This is likely an alarm setting
                if "INTENT" not in entities:
                    entities["INTENT"] = "set-alarm"
            elif any(word in text_lower for word in reminder_words):
                # This is likely a reminder setting
                if "INTENT" not in entities:
                    entities["INTENT"] = "set-reminder"
        
        # Special case: Extract QUERY when search-related words are present
        if "QUERY" not in entities:
            search_words = ["tìm", "tìm kiếm", "tra cứu", "search", "google"]
            if any(word in text_lower for word in search_words):
                # Try to extract everything after the search word
                for word in search_words:
                    if word in text_lower:
                        start_pos = text_lower.find(word) + len(word)
                        query = text[start_pos:].strip()
                        if query and len(query) > 3:  # Ensure it's not too short
                            entities["QUERY"] = query
                            break
        
        return entities
    
    def generate_value(self, intent: str, entities: Dict[str, str], original_text: str) -> str:
        if intent == "unknown" or intent == "error":
            return "Không thể xác định"
        
        # Tạo value dựa trên intent và entities
        if intent in ["call", "make-call", "make-video-call"]:
            receiver = entities.get("RECEIVER", "")
            if not receiver:
                # Try to extract receiver from text if not found by patterns
                potential_receivers = re.findall(r"(?:gọi|gọi cho|gọi điện cho|nhắn tin cho|gửi cho)\s+(\w+(?:\s+\w+){0,2})", original_text, re.IGNORECASE)
                if potential_receivers:
                    receiver = potential_receivers[0]
                else:
                    receiver = "người nhận"
            
            if intent == "make-video-call":
                return f"Gọi video cho {receiver}"
            else:
                return f"Gọi điện cho {receiver}"
        
        elif intent in ["send-mess", "send-message"]:
            message = entities.get("MESSAGE", "")
            receiver = entities.get("RECEIVER", "")
            
            # Xử lý trường hợp "Kiểm tra tin nhắn" bị phân loại sai thành "send-mess"
            if "kiểm tra" in original_text.lower() and "từ" in original_text.lower():
                match = re.search(r"từ\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if match:
                    from_person = match.group(1).strip()
                    return f"từ {from_person}"
            
            # Ưu tiên trích xuất nội dung chính từ MESSAGE entity
            if message:
                # Nếu message chứa "rằng là" hoặc tương tự, lấy phần sau đó
                if "rằng là" in message:
                    content = message.split("rằng là", 1)[-1].strip()
                    # Loại bỏ "là" ở đầu nếu có
                    if content.startswith("là "):
                        content = content[3:].strip()
                    return content
                elif message.startswith("là "):
                    # Nếu message bắt đầu bằng "là", loại bỏ nó
                    content = message[3:].strip()
                    return content
                elif "rằng" in message:
                    content = message.split("rằng", 1)[-1].strip()
                    # Loại bỏ "là" ở đầu nếu có
                    if content.startswith("là "):
                        content = content[3:].strip()
                    return content
                elif " là " in message:
                    # Xử lý trường hợp "anh Tuấn là đã nhận được tiền"
                    content = message.split(" là ", 1)[-1].strip()
                    return content
                else:
                    return message
            
            # Nếu không có MESSAGE entity, thử trích xuất từ text gốc
            else:
                # Pattern để tìm nội dung sau "rằng là" hoặc "rằng"
                patterns = [
                    r"rằng\s+là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:nhắn|gửi|nhắn tin|gửi tin nhắn)(?:\s+cho\s+\w+)?(?:\s+qua\s+\w+)?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])"
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, original_text, re.IGNORECASE)
                    if match:
                        content = match.group(1).strip()
                        # Loại bỏ các từ không cần thiết ở đầu
                        if content.startswith("là "):
                            content = content[3:].strip()
                        return content
                
                # Fallback
                if receiver:
                    return f"Tin nhắn cho {receiver}"
                else:
                    return "Nội dung tin nhắn"
        
        elif intent in ["set-alarm", "set-reminder"]:
            time_info = entities.get("TIME", "")
            
            if not time_info:
                # Try to extract time info from text
                time_patterns = [
                    r"(\d{1,2})\s*(?:giờ|h|:)\s*(\d{1,2})?\s*(?:phút)?",
                    r"(\d{1,2})\s*(?:giờ|h)\s*(?:rưỡi|buổi|sáng|trưa|chiều|tối)",
                    r"(\d{1,2})\s*(?:giờ|h)"
                ]
                
                for pattern in time_patterns:
                    match = re.search(pattern, original_text, re.IGNORECASE)
                    if match:
                        if match.group(2):  # Hour and minute
                            time_info = f"{match.group(1)}:{match.group(2)}"
                        else:  # Just hour
                            time_info = f"{match.group(1)}:00"
                        break
            
            # Check for period of day if we have time
            if time_info:
                period_match = re.search(r"(sáng|trưa|chiều|tối|đêm)", original_text, re.IGNORECASE)
                if period_match and period_match.group(1) not in time_info:
                    time_info = f"{time_info} {period_match.group(1)}"
            
            if intent == "set-alarm":
                description = "Báo thức"
                description_match = re.search(r"(?:báo thức|alarm)(?:\s+để)?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if description_match:
                    description = description_match.group(1).strip()
                
                return f"{time_info} - {description}" if time_info else "Thời gian báo thức"
            else:  # set-reminder
                # Cải thiện cho nhắc nhở thuốc
                description = "Nhắc nhở"
                
                # Tìm kiếm hành động cụ thể (đặc biệt là uống thuốc)
                medicine_patterns = [
                    r"uống\s+(\d+\s+)?(?:viên\s+)?(?:thuốc\s+)?(?:tiểu\s+đường|huyết\s+áp|tim|vitamin|sắt|cảm|đau\s+đầu|kháng\s+sinh)",
                    r"(?:thuốc\s+)?(?:tiểu\s+đường|huyết\s+áp|tim|vitamin|sắt|cảm|đau\s+đầu|kháng\s+sinh)",
                    r"uống\s+(.+?)(?:\s+(?:lúc|vào|sau|trước|nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:nhắc nhở|nhắc|reminder)(?:\s+về|về|về việc|việc)?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])"
                ]
                
                for pattern in medicine_patterns:
                    description_match = re.search(pattern, original_text, re.IGNORECASE)
                    if description_match:
                        if description_match.groups():
                            description = description_match.group(1).strip()
                        else:
                            description = description_match.group(0).strip()
                        break
                
                return f"{time_info} - {description}" if time_info else description
        
        elif intent == "check-weather":
            location = entities.get("LOCATION", "")
            time = entities.get("TIME", "")
            
            if not location:
                # Try to extract location from text
                location_match = re.search(r"(?:thời tiết|nhiệt độ|mưa|nắng)(?:\s+ở|tại|của)?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if location_match:
                    location = location_match.group(1).strip()
                else:
                    location = "khu vực hiện tại"
            
            if time:
                return f"{location} ({time})"
            else:
                return location
        
        elif intent == "check-device-status":
            device = entities.get("DEVICE", "thiết bị")
            return f"Kiểm tra trạng thái {device}"
        
        elif intent == "check-health-status":
            health_aspect = entities.get("HEALTH", "sức khỏe")
            return f"Kiểm tra {health_aspect}"
        
        elif intent == "check-messages":
            platform = entities.get("PLATFORM", "")
            receiver = entities.get("RECEIVER", "")
            
            # Ưu tiên trích xuất "từ ai" từ text gốc
            if "từ" in original_text.lower():
                match = re.search(r"từ\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if match:
                    from_person = match.group(1).strip()
                    return f"từ {from_person}"
            
            # Fallback với các entities có sẵn
            if receiver and receiver != "trên":  # "trên" không phải người gửi
                return f"từ {receiver}"
            elif platform:
                return f"Kiểm tra {platform}"
            else:
                return "Kiểm tra tin nhắn"
        
        elif intent in ["play-media", "play-audio", "play-content"]:
            content = entities.get("CONTENT", "")
            query = entities.get("QUERY", "")
            platform = entities.get("PLATFORM", "")
            
            if not content and not query:
                # Try to extract content from text with improved patterns
                content_patterns = [
                    r"(?:phát|mở|bật|nghe|xem)(?:\s+bài|nhạc|phim|video|clip)?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:bài hát|bài|ca khúc|nhạc|phim|video|clip)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:vào|mở)\s+(?:youtube|facebook|zalo|instagram|tiktok)\s+(?:để|để mà|mà)\s+(?:tìm|tìm kiếm|search|phát|nghe|xem)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:tìm kiếm|tìm|search)\s+(?:trên|qua|bằng|dùng)\s+(?:youtube|facebook|zalo|instagram|tiktok)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:danh sách|list|playlist)\s+(?:nhạc|music|video|clip|phim)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:nhạc|music|video|clip|phim)\s+(?:mới nhất|hot|trending|phổ biến)\s+(?:của|do|bởi)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])"
                ]
                
                for pattern in content_patterns:
                    match = re.search(pattern, original_text, re.IGNORECASE)
                    if match:
                        content = match.group(1).strip()
                        break
            
            # Use query if content is not available
            final_content = content if content else query
            
            if final_content:
                if platform:
                    return f"{final_content} trên {platform}"
                else:
                    return final_content
            else:
                return "Nội dung phát"
        
        elif intent in ["read-news", "read-content"]:
            content = entities.get("CONTENT", "")
            query = entities.get("QUERY", "")
            
            if not content and not query:
                # Try to extract content/topic from text
                content_match = re.search(r"(?:đọc|đọc tin|đọc báo|đọc tin tức)(?:\s+về|về)?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if content_match:
                    content = content_match.group(1).strip()
            
            if intent == "read-news":
                if content:
                    return f"Tin tức về {content}"
                elif query:
                    return f"Tin tức về {query}"
                else:
                    return "Tin tức"
            else:  # read-content
                if content:
                    return f"Đọc: {content}"
                elif query:
                    return f"Đọc về: {query}"
                else:
                    return "Nội dung đọc"
        
        elif intent == "view-content":
            content = entities.get("CONTENT", "")
            query = entities.get("QUERY", "")
            
            if content:
                return f"Xem: {content}"
            elif query:
                return f"Xem về: {query}"
            else:
                return "Nội dung xem"
        
        elif intent == "open-app":
            app = entities.get("APP", "")
            
            if not app:
                # Try to extract app name from text
                app_match = re.search(r"(?:mở|vào|khởi động|chạy|sử dụng|dùng)(?:\s+ứng dụng|app|phần mềm)?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if app_match:
                    app = app_match.group(1).strip()
                else:
                    app = "ứng dụng"
            
            return app
        
        elif intent == "open-app-action":
            app = entities.get("APP", "ứng dụng")
            action = entities.get("ACTION", "hành động")
            return f"{action} trong {app}"
        
        elif intent in ["search-content", "search-internet"]:
            query = entities.get("QUERY", "")
            platform = entities.get("PLATFORM", "")
            
            if not query:
                # Try to extract query from text with improved patterns
                query_patterns = [
                    r"(?:tìm|tìm kiếm|search|tra cứu|tra|kiếm|tìm hiểu)(?:\s+về|về|thông tin về)?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:tìm|tìm kiếm|search|tra cứu|tra|kiếm|tìm hiểu)(?:\s+(?:cho tôi|cho mình|cho bác|cho cô|cho chú|giúp tôi|giúp mình|giúp bác|giúp cô|giúp chú))?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:vào|mở)\s+(?:youtube|facebook|zalo|instagram|tiktok)\s+(?:để|để mà|mà)\s+(?:tìm|tìm kiếm|search|phát|nghe|xem)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:tìm kiếm|tìm|search)\s+(?:trên|qua|bằng|dùng)\s+(?:youtube|facebook|zalo|instagram|tiktok)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:danh sách|list|playlist)\s+(?:nhạc|music|video|clip|phim)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:nhạc|music|video|clip|phim)\s+(?:mới nhất|hot|trending|phổ biến)\s+(?:của|do|bởi)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])"
                ]
                
                for pattern in query_patterns:
                    match = re.search(pattern, original_text, re.IGNORECASE)
                    if match:
                        query = match.group(1).strip()
                        break
            
            if query:
                if platform:
                    return f"{query} trên {platform}"
                else:
                    return query
            else:
                return "Từ khóa tìm kiếm"
        
        elif intent == "browse-social-media":
            platform = entities.get("PLATFORM", "")
            
            if not platform:
                # Try to extract platform name from text
                platform_match = re.search(r"(?:lướt|duyệt|xem|vào|mở)(?:\s+(?:facebook|fb|zalo|instagram|tiktok|youtube|twitter))(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if platform_match:
                    platform = platform_match.group(1).strip()
                else:
                    platform = "mạng xã hội"
            
            return f"Duyệt {platform}"
        
        elif intent == "control-device":
            device = entities.get("DEVICE", "thiết bị")
            action = entities.get("ACTION", "")
            
            if not action:
                # Try to extract action from text
                action_match = re.search(r"(?:bật|tắt|mở|đóng|khóa|mở khóa|điều chỉnh|tăng|giảm|thay đổi)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if action_match:
                    action = action_match.group(0).strip()  # Use full match as the action
                else:
                    action = "điều khiển"
            
            return f"{action} {device}"
        
        elif intent == "adjust-settings":
            setting = entities.get("SETTING", "")
            
            if not setting:
                # Try to extract setting from text
                setting_match = re.search(r"(?:cài đặt|thiết lập|điều chỉnh|thay đổi|chỉnh|sửa)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if setting_match:
                    setting = setting_match.group(1).strip()
                else:
                    setting = "cài đặt"
            
            return f"Điều chỉnh {setting}"
        
        elif intent == "app-tutorial":
            app = entities.get("APP", "")
            
            if not app:
                # Try to extract app name from text
                app_match = re.search(r"(?:hướng dẫn|chỉ dẫn|chỉ|dạy|bày)(?:\s+(?:sử dụng|dùng|cách))?(?:\s+(?:ứng dụng|app|phần mềm))?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if app_match:
                    app = app_match.group(1).strip()
                else:
                    app = "ứng dụng"
            
            return f"Hướng dẫn sử dụng {app}"
        
        elif intent == "navigation-help":
            destination = entities.get("LOCATION", "")
            
            if not destination:
                # Try to extract destination from text
                destination_match = re.search(r"(?:đường|đường đi|chỉ đường|chỉ|đi|tới|đến|về)(?:\s+(?:tới|đến|về))?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if destination_match:
                    destination = destination_match.group(1).strip()
                else:
                    destination = "đích đến"
            
            return f"Điều hướng đến {destination}"
        
        elif intent == "provide-instructions":
            topic = entities.get("TOPIC", "")
            
            if not topic:
                # Try to extract topic from text
                topic_match = re.search(r"(?:hướng dẫn|chỉ dẫn|chỉ|dạy|bày)(?:\s+(?:về|cách))?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if topic_match:
                    topic = topic_match.group(1).strip()
                else:
                    topic = "chủ đề"
            
            return f"Hướng dẫn về {topic}"
        
        elif intent == "general-conversation":
            # Extract general conversational intent
            if "xin chào" in original_text.lower() or "hello" in original_text.lower() or "hi" in original_text.lower():
                return "Chào hỏi"
            elif "tạm biệt" in original_text.lower() or "bye" in original_text.lower():
                return "Tạm biệt"
            elif "cảm ơn" in original_text.lower() or "thanks" in original_text.lower() or "thank" in original_text.lower():
                return "Cảm ơn"
            elif "xin lỗi" in original_text.lower() or "sorry" in original_text.lower():
                return "Xin lỗi"
            elif "khỏe không" in original_text.lower() or "thế nào" in original_text.lower():
                return "Hỏi thăm"
            else:
                return "Trò chuyện thông thường"
        
        else:
            # Try to extract meaningful content from text for unknown intents
            if "tìm" in original_text.lower() or "tìm kiếm" in original_text.lower() or "search" in original_text.lower():
                # Extract search query
                search_patterns = [
                    r"(?:tìm|tìm kiếm|search)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:vào|mở)\s+\w+\s+(?:để|để mà|mà)\s+(?:tìm|tìm kiếm|search)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])"
                ]
                for pattern in search_patterns:
                    match = re.search(pattern, original_text, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()
            
            elif "phát" in original_text.lower() or "nghe" in original_text.lower() or "xem" in original_text.lower():
                # Extract media content
                media_patterns = [
                    r"(?:phát|nghe|xem)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:nhạc|video|phim)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])"
                ]
                for pattern in media_patterns:
                    match = re.search(pattern, original_text, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()
            
            elif "mở" in original_text.lower() or "vào" in original_text.lower():
                # Extract app/platform
                app_patterns = [
                    r"(?:mở|vào)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:ứng dụng|app)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])"
                ]
                for pattern in app_patterns:
                    match = re.search(pattern, original_text, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()
            
            # Fallback to generic message
            return f"Thực hiện hành động: {intent}"
    
    def predict_intent(self, text: str, confidence_threshold: float = 0.3) -> Dict:
        """Predict intent và confidence với GPU support"""
        try:
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=model_config.max_length,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            # Convert to FP16 nếu model đang sử dụng FP16
            if self.model.dtype == torch.float16:
                input_ids = input_ids.half()
                attention_mask = attention_mask.half()
            
            # Predict
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                predicted = torch.argmax(logits, dim=1)
                confidence = probabilities.max().item()
                intent = self.id_to_intent[predicted.item()]
            
            # Check confidence threshold
            if confidence < confidence_threshold:
                intent = "unknown"
                confidence = 0.0
            
            return {
                "intent": intent,
                "confidence": confidence,
                "probabilities": probabilities.cpu().numpy().tolist()[0]
            }
            
        except Exception as e:
            print(f"❌ Error predicting intent: {e}")
            return {
                "intent": "error",
                "confidence": 0.0,
                "probabilities": []
            }
    
    async def predict_with_reasoning(self, text: str) -> Dict[str, Any]:
        """Predict intent với reasoning engine cho các từ ngữ không có trong dataset - Cải tiến"""
        try:
            print(f"🧠 REASONING PREDICTION: '{text}'")
            start_time = datetime.now()
            
            # 1. Thử predict với model đã train trước (nếu có)
            try:
                model_result = None
                model_confidence = 0.0
                model_intent = "unknown"
                
                if self.model and self.tokenizer:
                    model_result = self.predict_intent(text)
                    model_confidence = model_result.get("confidence", 0.0)
                    model_intent = model_result.get("intent", "unknown")
                    print(f"🤖 MODEL PREDICTION: {model_intent} (confidence: {model_confidence:.3f})")
            except Exception as e:
                print(f"⚠️ Model prediction error: {str(e)}")
                model_result = {"intent": "unknown", "confidence": 0.0}
                model_confidence = 0.0
                model_intent = "unknown"
            
            # 2. Nếu confidence thấp hoặc intent là unknown, sử dụng reasoning
            if model_confidence < 0.6 or model_intent == "unknown":
                print("🔍 Confidence thấp hoặc intent unknown, sử dụng Reasoning Engine")
                
                # Kiểm tra từ khóa nhắn tin trước khi dùng reasoning
                text_lower = text.lower()
                message_keywords = ["nhắn tin", "gửi tin", "soạn tin", "text", "sms", "message", "gửi", "nhắn"]
                has_message_keyword = any(keyword in text_lower for keyword in message_keywords)
                
                reasoning_result = self.reasoning_engine.reasoning_predict(text)
                reasoning_intent = reasoning_result.get("intent", "unknown")
                reasoning_confidence = reasoning_result.get("confidence", 0.0)
                print(f"🧠 REASONING PREDICTION: {reasoning_intent} (confidence: {reasoning_confidence:.3f})")
                
                # Ưu tiên call nếu có từ khóa gọi điện
                call_keywords = ["cuộc gọi", "gọi thoại", "gọi điện", "thực hiện gọi", "thực hiện cuộc gọi"]
                has_call_keyword = any(keyword in text_lower for keyword in call_keywords)
                
                if has_call_keyword and reasoning_intent != "call":
                    print("🔧 Override intent to call due to call keywords")
                    reasoning_intent = "call"
                    reasoning_confidence = max(reasoning_confidence, 0.8)  # Boost confidence
                
                # Ưu tiên send-mess nếu có từ khóa nhắn tin
                elif has_message_keyword and reasoning_intent != "send-mess":
                    print("🔧 Override intent to send-mess due to message keywords")
                    reasoning_intent = "send-mess"
                    reasoning_confidence = max(reasoning_confidence, 0.7)  # Boost confidence
                
                # Decide which intent to use (model or reasoning)
                final_intent = reasoning_intent
                final_confidence = reasoning_confidence
                method = "reasoning_engine"
                
                # If model had a non-unknown prediction with reasonable confidence, compare
                if model_intent != "unknown" and model_confidence >= 0.4:
                    # Check if model and reasoning agree
                    if model_intent == reasoning_intent:
                        # Both agree, boost confidence
                        final_confidence = max(model_confidence, reasoning_confidence) + 0.1
                        final_confidence = min(final_confidence, 0.99)  # Cap at 0.99
                        method = "model_reasoning_agreement"
                    else:
                        # They disagree, use the one with higher confidence
                        if model_confidence > reasoning_confidence + 0.2:  # Model significantly more confident
                            final_intent = model_intent
                            final_confidence = model_confidence
                            method = "model_override"
                        # Otherwise, stick with reasoning (default above)
            else:
                # Model has high confidence, use its prediction
                final_intent = model_intent
                final_confidence = model_confidence
                method = "trained_model"
                reasoning_result = None
            
            # Extract entities using our improved method
            try:
                entities = self.extract_entities(text)
            except Exception as e:
                print(f"⚠️ Error extracting entities: {e}")
                entities = {}
            
            # Get command from intent
            command = self.intent_to_command.get(final_intent, "unknown")
            
            # Generate value from intent and entities
            try:
                value = self.generate_value(final_intent, entities, text)
            except Exception as e:
                print(f"⚠️ Error generating value: {e}")
                value = f"Thực hiện hành động: {final_intent}"
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Construct and return the final result
            result = {
                "text": text,
                "intent": final_intent,
                "confidence": final_confidence,
                "command": command,
                "entities": entities,
                "value": value,
                "method": method,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add reasoning details if available (simplified)
            if reasoning_result:
                result["reasoning_details"] = {
                    "semantic_similarity": reasoning_result.get("semantic_similarity", {})
                }
            
            return result
                    
        except Exception as e:
            print(f"❌ Error in reasoning prediction: {str(e)}")
            import traceback
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            return {
                "text": text,
                "intent": "unknown",
                "confidence": 0.0,
                "command": "unknown",
                "entities": {},
                "value": "",
                "method": "error",
                "error": str(e),
                "processing_time": 0.0,
                "timestamp": datetime.now().isoformat()
            }
    
    async def batch_predict_with_reasoning(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch predict với reasoning engine"""
        results = []
        for text in texts:
            result = await self.predict_with_reasoning(text)
            results.append(result)
        return results
    
    async def analyze_text_semantics(self, text: str) -> Dict[str, Any]:
        """Phân tích semantic của text"""
        try:
            # Lấy embedding
            embedding = self.reasoning_engine.get_text_embedding(text)
            
            # Tìm similar intents
            similar_intents = self.reasoning_engine.find_similar_intents(text)
            
            # Extract context features
            context_features = self.reasoning_engine.extract_context_features(text)
            
            # Pattern matching
            pattern_results = self.reasoning_engine.pattern_matching(text)
            
            # Keyword matching
            keyword_results = self.reasoning_engine.keyword_matching(text)
            
            return {
                "text": text,
                "embedding_shape": embedding.shape,
                "similar_intents": similar_intents,
                "context_features": context_features,
                "pattern_matching": pattern_results,
                "keyword_matching": keyword_results,
                "semantic_analysis": {
                    "has_time_context": context_features["has_time"],
                    "has_person_context": context_features["has_person"],
                    "has_action_context": context_features["has_action"],
                    "has_object_context": context_features["has_object"]
                }
            }
            
        except Exception as e:
            return {
                "text": text,
                "error": str(e)
            }
    
    async def update_knowledge_base(self, new_patterns: Dict[str, List[str]]) -> Dict[str, Any]:
        """Cập nhật knowledge base của reasoning engine"""
        try:
            self.reasoning_engine.update_knowledge_base(new_patterns)
            return {
                "status": "success",
                "message": "Knowledge base updated successfully",
                "updated_patterns": new_patterns
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def save_knowledge_base(self, file_path: str = "knowledge_base.json") -> Dict[str, Any]:
        """Lưu knowledge base"""
        try:
            self.reasoning_engine.save_knowledge_base(file_path)
            return {
                "status": "success",
                "message": f"Knowledge base saved to {file_path}",
                "file_path": file_path
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def load_knowledge_base(self, file_path: str = "knowledge_base.json") -> Dict[str, Any]:
        """Load knowledge base"""
        try:
            self.reasoning_engine.load_knowledge_base(file_path)
            return {
                "status": "success",
                "message": f"Knowledge base loaded from {file_path}",
                "file_path": file_path
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def process_text(self, text: str, confidence_threshold: float = 0.3) -> IntentResponse:
        """Xử lý text và trả về kết quả đầy đủ"""
        start_time = datetime.now()
        
        # Predict intent - không cần await vì predict_intent không phải async
        intent_result = self.predict_intent(text, confidence_threshold)
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Get command
        command = self.intent_to_command.get(intent_result["intent"], "unknown")
        
        # Generate value based on intent and entities
        value = self.generate_value(intent_result["intent"], entities, text)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return IntentResponse(
            input_text=text,
            intent=intent_result["intent"],
            confidence=intent_result["confidence"],
            command=command,
            entities=entities,
            value=value,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )

# Initialize API
api = PhoBERT_SAM_API()

# FastAPI app
app = FastAPI(
    title="PhoBERT_SAM API",
    description="API cho Intent Recognition và Entity Extraction cho người cao tuổi",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origins (có thể thay đổi thành specific domains)
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả HTTP methods
    allow_headers=["*"],  # Cho phép tất cả headers
)

@app.on_event("startup")
async def startup_event():
    """Khởi tạo model khi server start"""
    if not api.load_model():
        raise Exception("Không thể load model!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PhoBERT_SAM API",
        "version": "1.0.0",
        "status": "running",
        "available_intents": list(api.intent_to_command.keys()) if api.id_to_intent else []
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": api.model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_intent(request: IntentRequest):
    """Predict intent và extract entities - Simplified response"""
    try:
        # Nếu model không có, sử dụng reasoning engine
        if not api.model:
            reasoning_result = await api.predict_with_reasoning(request.text)
            
            # Return simplified response
            return {
                "text": request.text,
                "intent": reasoning_result["intent"],
                "confidence": reasoning_result["confidence"],
                "command": reasoning_result["command"],
                "entities": reasoning_result["entities"],
                "value": reasoning_result["value"],
                "method": reasoning_result.get("method", "reasoning_engine"),
                "processing_time": reasoning_result["processing_time"],
                "timestamp": reasoning_result["timestamp"],
                "reasoning_details": reasoning_result.get("reasoning_details", {})
            }
        
        # Nếu có model, sử dụng process_text
        result = await api.process_text(request.text, request.confidence_threshold)
        
        # Convert to simplified format
        return {
            "text": result.input_text,
            "intent": result.intent,
            "confidence": result.confidence,
            "command": result.command,
            "entities": result.entities,
            "value": result.value,
            "method": "trained_model",
            "processing_time": result.processing_time,
            "timestamp": result.timestamp
        }
        
    except Exception as e:
        print(f"❌ Error in predict endpoint: {str(e)}")
        
        # Return simplified error response
        return {
            "text": request.text,
            "intent": "unknown",
            "confidence": 0.0,
            "command": "unknown",
            "entities": {},
            "value": "",
            "method": "error",
            "processing_time": 0.0,
            "timestamp": datetime.now().isoformat()
        }

@app.get("/intents")
async def get_intents():
    """Lấy danh sách intents có sẵn"""
    if not api.id_to_intent:
        raise HTTPException(status_code=500, detail="Model chưa được load")
    
    return {
        "intents": list(api.id_to_intent.values()),
        "intent_to_command": api.intent_to_command
    }

@app.get("/entities")
async def get_entity_patterns():
    """Lấy patterns cho entity extraction"""
    return {
        "entity_patterns": api.entity_patterns
    }

@app.post("/batch_predict")
async def batch_predict(texts: List[str], confidence_threshold: float = 0.3):
    """Predict nhiều texts cùng lúc"""
    try:
        if not api.model:
            raise HTTPException(status_code=500, detail="Model chưa được load")
        
        results = []
        for text in texts:
            result = await api.process_text(text, confidence_threshold)
            results.append(result.dict())
        
        return {
            "results": results,
            "total_processed": len(texts)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý batch: {str(e)}")

# Thêm các endpoints mới cho reasoning
@app.post("/predict-with-reasoning")
async def predict_with_reasoning(request: Dict[str, Any]):
    """Predict intent với reasoning engine"""
    text = request.get("text", "")
    if not text:
        return {"error": "Text is required"}
    
    result = await api.predict_with_reasoning(text)
    return result

@app.post("/batch-predict-with-reasoning")
async def batch_predict_with_reasoning(request: Dict[str, Any]):
    """Batch predict với reasoning engine"""
    texts = request.get("texts", [])
    if not texts:
        return {"error": "Texts list is required"}
    
    results = await api.batch_predict_with_reasoning(texts)
    return {"results": results}

@app.post("/analyze-semantics")
async def analyze_semantics(request: Dict[str, Any]):
    """Phân tích semantic của text"""
    text = request.get("text", "")
    if not text:
        return {"error": "Text is required"}
    
    result = await api.analyze_text_semantics(text)
    return result

@app.post("/update-knowledge-base")
async def update_knowledge_base(request: Dict[str, Any]):
    """Cập nhật knowledge base"""
    new_patterns = request.get("patterns", {})
    if not new_patterns:
        return {"error": "Patterns are required"}
    
    result = await api.update_knowledge_base(new_patterns)
    return result

@app.post("/save-knowledge-base")
async def save_knowledge_base(request: Dict[str, Any]):
    """Lưu knowledge base"""
    file_path = request.get("file_path", "knowledge_base.json")
    result = await api.save_knowledge_base(file_path)
    return result

@app.post("/load-knowledge-base")
async def load_knowledge_base(request: Dict[str, Any]):
    """Load knowledge base"""
    file_path = request.get("file_path", "knowledge_base.json")
    result = await api.load_knowledge_base(file_path)
    return result

if __name__ == "__main__":
    print("🚀 Starting PhoBERT_SAM API Server...")
    print("=" * 50)
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("🎯 Predict Endpoint: POST http://localhost:8000/predict")
    print("=" * 50)
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
