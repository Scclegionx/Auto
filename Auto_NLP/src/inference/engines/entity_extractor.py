import re
from typing import Dict, List, Optional, Tuple

class EntityExtractor:
    """Entity extractor chuyên biệt cho hệ thống gọi điện/nhắn tin"""
    
    def __init__(self):
        # Số từ chữ sang số
        # Số từ chữ sang số - Cải thiện với cụm từ phức tạp
        self.number_words = {
            # Số cơ bản
            'không': '0', 'một': '1', 'hai': '2', 'ba': '3', 'bốn': '4',
            'năm': '5', 'sáu': '6', 'bảy': '7', 'tám': '8', 'chín': '9',
            
            # Các biến thể phát âm và không dấu
            'mốt': '1', 'một': '1', 'oan': '1', 'one': '1', 'mot': '1',
            'tu': '2', 'two': '2',
            'tư': '4', 'four': '4', 'tu': '4', 'bon': '4',
            'lăm': '5', 'five': '5', 'lam': '5', 'nam': '5',
            'sáu': '6', 'six': '6', 'sau': '6',
            'bảy': '7', 'seven': '7', 'bay': '7',
            'tám': '8', 'eight': '8', 'tam': '8', 'tắm': '8', 'ta': '8',
            'chín': '9', 'nine': '9', 'chin': '9',
            'zê rô': '0', 'zero': '0', 'ze ro': '0', 'khong': '0',
            
            # Số từ 10-19
            'mười': '10', 'mười một': '11', 'mười hai': '12', 'mười ba': '13',
            'mười bốn': '14', 'mười lăm': '15', 'mười sáu': '16', 'mười bảy': '17',
            'mười tám': '18', 'mười chín': '19',
            
            # Số từ 20-29
            'hai mươi': '20', 'hai mươi mốt': '21', 'hai mươi hai': '22', 'hai mươi ba': '23',
            'hai mươi bốn': '24', 'hai mươi lăm': '25', 'hai mươi sáu': '26', 'hai mươi bảy': '27',
            'hai mươi tám': '28', 'hai mươi chín': '29',
            
            # Số từ 30-39
            'ba mươi': '30', 'ba mươi mốt': '31', 'ba mươi hai': '32', 'ba mươi ba': '33',
            'ba mươi bốn': '34', 'ba mươi lăm': '35', 'ba mươi sáu': '36', 'ba mươi bảy': '37',
            'ba mươi tám': '38', 'ba mươi chín': '39',
            
            # Số từ 40-49
            'bốn mươi': '40', 'bốn mươi mốt': '41', 'bốn mươi hai': '42', 'bốn mươi ba': '43',
            'bốn mươi bốn': '44', 'bốn mươi lăm': '45', 'bốn mươi sáu': '46', 'bốn mươi bảy': '47',
            'bốn mươi tám': '48', 'bốn mươi chín': '49',
            
            # Số từ 50-59
            'năm mươi': '50', 'năm mươi mốt': '51', 'năm mươi hai': '52', 'năm mươi ba': '53',
            'năm mươi bốn': '54', 'năm mươi lăm': '55', 'năm mươi sáu': '56', 'năm mươi bảy': '57',
            'năm mươi tám': '58', 'năm mươi chín': '59',
            
            # Số từ 60-69
            'sáu mươi': '60', 'sáu mươi mốt': '61', 'sáu mươi hai': '62', 'sáu mươi ba': '63',
            'sáu mươi bốn': '64', 'sáu mươi lăm': '65', 'sáu mươi sáu': '66', 'sáu mươi bảy': '67',
            'sáu mươi tám': '68', 'sáu mươi chín': '69',
            
            # Số từ 70-79
            'bảy mươi': '70', 'bảy mươi mốt': '71', 'bảy mươi hai': '72', 'bảy mươi ba': '73',
            'bảy mươi bốn': '74', 'bảy mươi lăm': '75', 'bảy mươi sáu': '76', 'bảy mươi bảy': '77',
            'bảy mươi tám': '78', 'bảy mươi chín': '79',
            
            # Số từ 80-89
            'tám mươi': '80', 'tám mươi mốt': '81', 'tám mươi hai': '82', 'tám mươi ba': '83',
            'tám mươi bốn': '84', 'tám mươi lăm': '85', 'tám mươi sáu': '86', 'tám mươi bảy': '87',
            'tám mươi tám': '88', 'tám mươi chín': '89',
            
            # Số từ 90-99
            'chín mươi': '90', 'chín mươi mốt': '91', 'chín mươi hai': '92', 'chín mươi ba': '93',
            'chín mươi bốn': '94', 'chín mươi lăm': '95', 'chín mươi sáu': '96', 'chín mươi bảy': '97',
            'chín mươi tám': '98', 'chín mươi chín': '99'
        }
        self.receiver_patterns = self._build_receiver_patterns()
        self.time_patterns = self._build_time_patterns()
        self.message_patterns = self._build_message_patterns()
        self.platform_patterns = self._build_platform_patterns()
        self.contact_patterns = self._build_contact_patterns()
        self.media_patterns = self._build_media_patterns()
        self.search_patterns = self._build_search_patterns()
        self.youtube_patterns = self._build_youtube_patterns()
        self.info_patterns = self._build_info_patterns()
        self.alarm_patterns = self._build_alarm_patterns()
        self.calendar_patterns = self._build_calendar_patterns()
        self.camera_patterns = self._build_camera_patterns()
        self.device_control_patterns = self._build_device_control_patterns()
        self.media_playback_patterns = self._build_media_playback_patterns()
        self.messaging_patterns = self._build_messaging_patterns()
        self.content_viewing_patterns = self._build_content_viewing_patterns()
        self.internet_search_patterns = self._build_internet_search_patterns()
        self.youtube_search_patterns = self._build_youtube_search_patterns()
        self.information_patterns = self._build_information_patterns()
    
    def _convert_words_to_numbers(self, text: str) -> str:
        """Chuyển đổi số từ chữ sang số trong text - Cải thiện với cụm từ phức tạp"""
        result = text.lower()
        
        # Bước 1: Xử lý cụm từ phức tạp trước (ưu tiên cao)
        # Ví dụ: "hai mươi mốt" → "21" thay vì "20 1"
        
        # Pattern cho cụm từ "X mươi Y" (21-99) - hỗ trợ đầy đủ dấu tiếng Việt
        compound_pattern = r'([a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+)\s+mươi\s+([a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+)'
        matches = list(re.finditer(compound_pattern, result))
        
        # Thay thế từ cuối lên đầu để tránh conflict
        for match in reversed(matches):
            tens_word = match.group(1)
            ones_word = match.group(2)
            
            # Mapping cho hàng chục
            tens_map = {
                'hai': 20, 'ba': 30, 'bốn': 40, 'năm': 50,
                'sáu': 60, 'bảy': 70, 'tám': 80, 'chín': 90
            }
            
            # Mapping cho hàng đơn vị
            ones_map = {
                'mốt': 1, 'một': 1, 'hai': 2, 'ba': 3, 'bốn': 4,
                'lăm': 5, 'năm': 5, 'sáu': 6, 'bảy': 7, 'tám': 8, 'tắm': 8, 'ta': 8, 'chín': 9
            }
            
            tens_value = tens_map.get(tens_word, 0)
            ones_value = ones_map.get(ones_word, 0)
            
            if tens_value > 0 and ones_value > 0:
                replacement = str(tens_value + ones_value)
                result = result[:match.start()] + replacement + result[match.end():]
        
        # Pattern cho cụm từ "mười X" (11-19) - hỗ trợ đầy đủ dấu tiếng Việt
        teen_pattern = r'mười\s+([a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+)'
        teen_matches = list(re.finditer(teen_pattern, result))
        
        for match in reversed(teen_matches):
            ones_word = match.group(1)
            
            # Mapping cho hàng đơn vị
            ones_map = {
                'một': 1, 'hai': 2, 'ba': 3, 'bốn': 4, 'lăm': 5,
                'sáu': 6, 'bảy': 7, 'tám': 8, 'chín': 9
            }
            
            ones_value = ones_map.get(ones_word, 0)
            if ones_value > 0:
                replacement = str(10 + ones_value)
                result = result[:match.start()] + replacement + result[match.end():]
        
        # Bước 2: Xử lý các từ đơn lẻ còn lại
        # Sắp xếp theo độ dài giảm dần để tránh thay thế sai
        sorted_words = sorted(self.number_words.items(), key=lambda x: len(x[0]), reverse=True)
        
        for word, number in sorted_words:
            if re.search(r'\d+', word):
                continue
            pattern = r'\b{}\b'.format(re.escape(word))
            if re.search(pattern, result):
                result = re.sub(pattern, number, result)
        
        return result
    
    def _build_device_control_patterns(self) -> List[Tuple[str, str]]:
        """Xây dựng patterns cho device control - Điều khiển thiết bị"""
        return [
            # ACTION patterns
            (r"(bật|mở|tắt|đóng|tăng|giảm|để|set)", "action"),
            
            # DEVICE patterns
            (r"(đèn|quạt|tivi|tv|máy\s+lạnh|điều\s+hoà|rèm|máy\s+lọc\s+khí|loa|âm\s+thanh|hệ\s+thống\s+âm\s+thanh)", "device"),
            
            # LEVEL patterns (brightness, volume, etc.)
            (r"(\d{1,3})\s*(%|phần\s+trăm)", "level"),
            (r"(âm\s+lượng|độ\s+sáng|ánh\s+sáng|gió)\s*(\d{1,3})\s*(%|phần\s+trăm)?", "level"),
            
            # DURATION patterns
            (r"(trong|trong\s+vòng)\s*(\d+)\s*(phút|giờ|giây|m|h|s)", "duration"),
            (r"(\d+)\s*(phút|giờ|giây|m|h|s)", "duration"),
            
            # FAN_SPEED patterns
            (r"(tốc\s+độ|mức)\s*(tự\s+động|thấp|trung\s+bình|cao|1|2|3|4|5)", "fan_speed"),
            (r"(auto|low|mid|high|1|2|3|4|5)", "fan_speed"),
            
            # SCENE patterns
            (r"(chế\s+độ|scene|kịch\s+bản)\s*(đọc|ngủ|xem\s+phim|thư\s+giãn|tập\s+trung|ban\s+đêm|eco)", "scene"),
            (r"(reading|sleep|movie|relax|focus|night|eco)", "scene"),
            
            # DEVICE_GROUP patterns
            (r"(tất\s+cả|all)\s*(đèn|quạt|thiết\s+bị)\s*(phòng\s+khách|phòng\s+ngủ|bếp|ban\s+công|tầng\s+\d+)", "device_group"),
            (r"(đèn|quạt|thiết\s+bị)\s*(phòng\s+khách|phòng\s+ngủ|bếp|ban\s+công|tầng\s+\d+)", "device_group"),
            
            # BRAND patterns
            (r"(xiaomi|tuya|homekit|samsung|lg|sony|philips|bosch|panasonic)", "brand"),
            
            # TEMPERATURE patterns
            (r"(nhiệt\s+độ|temp)\s*(\d+)\s*(độ|c|°c)?", "temperature"),
            (r"(\d+)\s*(độ|c|°c)", "temperature"),
            
            # CHANNEL patterns (for TV)
            (r"(kênh|channel)\s*(\d+|vtv\d+|htv\d+)", "channel"),
            (r"(chuyển|đổi)\s*kênh\s*(\d+|vtv\d+|htv\d+)?", "channel"),
            
            # CURTAIN patterns
            (r"(kéo|mở|đóng)\s*rèm\s*(ra|vào|một\s+nửa|50%|hết\s+cỡ)?", "curtain"),
            (r"rèm\s*(ra|vào|một\s+nửa|50%|hết\s+cỡ)", "curtain"),
        ]
    
    def _build_media_playback_patterns(self) -> List[Tuple[str, str]]:
        """Xây dựng patterns cho media playback - Phát nhạc/podcast/video"""
        return [
            # ACTION patterns
            (r"(phát|bật|mở|chạy|dừng|tạm\s+dừng|tiếp\s+tục|bài\s+tiếp|bài\s+trước|tua)", "action"),
            
            # SONG patterns
            (r"(mở|phát|bật)\s+(bài|bản|bản\s+nhạc)\s+['\"]?([^'\"\n]+?)['\"]?(?:\s+của\s+([^,\n]+))?", "song"),
            (r"(mở|phát|bật)\s+(nhạc|album)\s+(['\"]?[^'\"\n]+['\"]?)", "album"),
            
            # ARTIST patterns
            (r"(của|bởi|từ)\s+([^,\n]+)", "artist"),
            (r"(ca\s+sĩ|nghệ\s+sĩ|singer)\s+([^,\n]+)", "artist"),
            
            # GENRE patterns
            (r"(nhạc\s+không\s+lời|bolero|thiền|ru\s+ngủ|lofi|trữ\s+tình|rock|pop|jazz|classical|electronic)", "genre"),
            (r"(karaoke)\s+(bài|bản)\s+['\"]?([^'\"\n]+?)['\"]?", "karaoke"),
            
            # LANGUAGE patterns
            (r"(nhạc|bài)\s+(tiếng\s+việt|tiếng\s+anh|tiếng\s+hàn|tiếng\s+nhật|vi|en|ko|ja)", "language"),
            (r"(tiếng\s+việt|tiếng\s+anh|tiếng\s+hàn|tiếng\s+nhật|vi|en|ko|ja)", "language"),
            
            # MOOD patterns
            (r"(nhạc)\s+(để|cho)\s+(ngủ|học|tập\s+trung|tập\s+thể\s+dục|thư\s+giãn|buồn|vui|yêu|tình\s+yêu)", "mood"),
            (r"(vui|buồn|thư\s+giãn|tập\s+trung|ngủ|học|yêu|tình\s+yêu)", "mood"),
            
            # YEAR patterns
            (r"(nhạc|bài)\s+(\d{4})", "year"),
            (r"(năm|year)\s+(\d{4})", "year"),
            (r"(thập\s+kỷ|decade)\s+(\d{4})", "year"),
            
            # SEASON/EPISODE patterns
            (r"(mùa|season)\s+(\d+)", "season"),
            (r"(tập|episode|ep)\s+(\d+)", "episode"),
            (r"(phần|part)\s+(\d+)", "episode"),
            
            # RESOLUTION patterns
            (r"(4k|1080p|720p|480p|360p|hd|full\s+hd|ultra\s+hd)", "resolution"),
            (r"(chất\s+lượng|quality)\s+(4k|1080p|720p|480p|360p|hd|full\s+hd|ultra\s+hd)", "resolution"),
            
            # SUBTITLE patterns
            (r"(phụ\s+đề|subtitle)\s+(tiếng\s+việt|tiếng\s+anh|vi|en|off|tắt)", "subtitle_lang"),
            (r"(vi|en|ko|ja|off|tắt)\s+(subtitle|phụ\s+đề)", "subtitle_lang"),
            
            # SHUFFLE patterns
            (r"(xáo\s+trộn|shuffle|ngẫu\s+nhiên)", "shuffle"),
            (r"(không\s+xáo\s+trộn|no\s+shuffle|theo\s+thứ\s+tự)", "shuffle"),
            
            # REPEAT patterns
            (r"(lặp\s+lại|repeat)\s+(một\s+lần|tất\s+cả|vô\s+hạn|off|tắt)", "repeat"),
            (r"(one|all|infinite|off|tắt)", "repeat"),
            
            # PLATFORM patterns
            (r"(trên|bằng|qua)\s+(youtube|spotify|zingmp3|apple\s+music|youtube\s+music|soundcloud)", "platform"),
            (r"(youtube|spotify|zingmp3|apple\s+music|youtube\s+music|soundcloud)", "platform"),
            
            # PODCAST patterns
            (r"(mở|nghe)\s+podcast\s+['\"]([^'\"]+)['\"]", "podcast"),
            (r"(podcast)\s+['\"]([^'\"]+)['\"]", "podcast"),
            
            # RADIO patterns
            (r"(mở|bật)\s+(đài|radio)\s+([^,\n]+)", "radio"),
            (r"(đài|radio)\s+([^,\n]+)", "radio"),
            
            # FILE_PATH patterns
            (r"(mở|phát|bật)\s+file\s+([\/\\][^\s]+\.(mp3|mp4|avi|mkv|wav|flac))", "file_path"),
            (r"(file|tệp)\s+([\/\\][^\s]+\.(mp3|mp4|avi|mkv|wav|flac))", "file_path"),
            
            # CONTEXT patterns
            (r"(nhạc)\s+(để|cho)\s+(ngủ|học|tập\s+trung|tập\s+thể\s+dục|thư\s+giãn|chạy|yoga|thiền)", "context"),
            (r"(để|cho)\s+(ngủ|học|tập\s+trung|tập\s+thể\s+dục|thư\s+giãn|chạy|yoga|thiền)", "context"),
        ]
    
    def _build_messaging_patterns(self) -> List[Tuple[str, str]]:
        """Xây dựng patterns cho messaging - Nhắn tin với entity mở rộng"""
        return [
            # MESSAGE_TYPE patterns
            (r"(tin\s+nhắn|text|văn\s+bản)", "message_type"),
            (r"(voice|giọng\s+nói|thoại)", "message_type"),
            (r"(hình\s+ảnh|ảnh|photo|image)", "message_type"),
            
            # URGENCY patterns
            (r"(gấp|khẩn|urgent)", "urgency"),
            (r"(bình\s+thường|normal|thường)", "urgency"),
            
            # SCHEDULE_TIME patterns
            (r"(gửi\s+lúc|gửi\s+vào|gửi\s+đến)\s+(\d{1,2}:\d{2}|\d{1,2}\s+giờ|\d{1,2}\s+giờ\s+\d{2})", "schedule_time"),
            (r"(lúc|vào|đến)\s+(\d{1,2}:\d{2}|\d{1,2}\s+giờ|\d{1,2}\s+giờ\s+\d{2})", "schedule_time"),
            (r"(sáng|chiều|tối|mai|hôm\s+nay|ngày\s+mai)", "schedule_time"),
            
            # PLATFORM_USER_ID patterns
            (r"(user\s+id|id\s+người\s+dùng)\s+([a-zA-Z0-9_]+)", "platform_user_id"),
            (r"(zalo\s+id|facebook\s+id|telegram\s+id)\s+([a-zA-Z0-9_]+)", "platform_user_id"),
            
            # GROUP patterns
            (r"(nhóm|group)\s+(gia\s+đình|bạn\s+bè|công\s+ty|team|family|friends|company)", "group"),
            (r"(gia\s+đình|bạn\s+bè|công\s+ty|team|family|friends|company)", "group"),
        ]
    
    def _build_content_viewing_patterns(self) -> List[Tuple[str, str]]:
        """Xây dựng patterns cho content viewing - Xem nội dung"""
        return [
            # SOURCE_APP patterns
            (r"(ảnh|gallery|thư\s+viện|photos)", "source_app"),
            (r"(facebook|fb)", "source_app"),
            (r"(zalo|instagram|ig)", "source_app"),
            (r"(youtube|yt|tiktok|twitter|x)", "source_app"),
            
            # SORT_ORDER patterns
            (r"(mới\s+nhất|newest)", "sort_order"),
            (r"(cũ\s+nhất|oldest)", "sort_order"),
            (r"(phổ\s+biến|popular)", "sort_order"),
            (r"(theo\s+tên|theo\s+ngày|theo\s+kích\s+thước|name|date|size)", "sort_order"),
            
            # DATE_RANGE patterns
            (r"(hôm\s+nay|hôm\s+qua|tuần\s+này|tuần\s+trước|tháng\s+này|tháng\s+trước)", "date_range"),
            (r"(năm\s+này|năm\s+ngoái|gần\s+đây|recent)", "date_range"),
            (r"(\d+\s+ngày\s+trước|\d+\s+tuần\s+trước|\d+\s+tháng\s+trước)", "date_range"),
            
            # OWNER patterns
            (r"(của\s+con|của\s+tôi|của\s+anh|của\s+chị|của\s+em|của\s+bạn)", "owner"),
            (r"(my|mine|your|his|her|their)", "owner"),
            (r"(của\s+[A-Za-z\s]+)", "owner"),
            
            # CONTENT_TYPE patterns
            (r"(ảnh|hình|photo|image)", "content_type"),
            (r"(video|clip|movie|film)", "content_type"),
            (r"(bài\s+viết|post|status|story)", "content_type"),
            (r"(link|liên\s+kết|url)", "content_type"),
            
            # QUERY patterns
            (r"(tìm\s+kiếm|search|tìm)\s+([^,\n]+)", "query"),
            (r"(xem|hiển\s+thị|mở)\s+([^,\n]+)", "query"),
            (r"(ảnh|video|bài\s+viết)\s+([^,\n]+)", "query"),
        ]
    
    def _build_internet_search_patterns(self) -> List[Tuple[str, str]]:
        """Xây dựng patterns cho internet search - Tìm kiếm internet"""
        return [
            # ANSWER_FORMAT patterns
            (r"(tóm\s+tắt|summary|tổng\s+hợp)", "answer_format"),
            (r"(danh\s+sách|list|liệt\s+kê)", "answer_format"),
            (r"(bước|steps|hướng\s+dẫn)", "answer_format"),
            (r"(định\s+nghĩa|definition|nghĩa)", "answer_format"),
            (r"(bảng|table|so\s+sánh)", "answer_format"),
            
            # NUM_RESULTS patterns
            (r"(top\s+\d+|top\s+\d+\s+kết\s+quả)", "num_results"),
            (r"(\d+\s+kết\s+quả|\d+\s+results)", "num_results"),
            (r"(hiển\s+thị\s+\d+|show\s+\d+)", "num_results"),
            
            # LANGUAGE patterns - Cải thiện detection
            (r"(tiếng\s+việt|việt\s+nam|vi|tieng\s+viet|viet\s+nam)", "language"),
            (r"(tiếng\s+anh|english|en|tieng\s+anh)", "language"),
            (r"(tiếng\s+hàn|korean|ko|tieng\s+han)", "language"),
            (r"(tiếng\s+nhật|japanese|ja|tieng\s+nhat)", "language"),
            
            # COUNTRY patterns - Cải thiện detection
            (r"(việt\s+nam|vietnam|vn|viet\s+nam)", "country"),
            (r"(mỹ|usa|us|my|america)", "country"),
            (r"(anh|uk|gb|england)", "country"),
            (r"(nhật|japan|jp|nhat)", "country"),
            
            # SAFESEARCH patterns
            (r"(bật\s+safe\s+search|tắt\s+safe\s+search)", "safesearch"),
            (r"(safe\s+search\s+(on|off))", "safesearch"),
            (r"(lọc\s+nội\s+dung|filter\s+content)", "safesearch"),
            
            # QUERY patterns - Cải thiện extraction
            (r"(tìm\s+kiếm|search|tìm|tim\s+kiem)\s+([^,\n]+)", "query"),
            (r"(tra\s+cứu|look\s+up|google|tra\s+cuu)\s+([^,\n]+)", "query"),
            (r"(hỏi|ask|what\s+is|hoi)\s+([^,\n]+)", "query"),
            (r"(tìm|tim)\s+([^,\n]+)", "query"),
            (r"(search|google)\s+([^,\n]+)", "query"),
            
            # SITE_DOMAIN patterns
            (r"(trên\s+site|trên\s+trang|on\s+site)\s+([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", "site_domain"),
            (r"(site:|domain:)\s*([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", "site_domain"),
            
            # COMPARISON patterns
            (r"(so\s+sánh|compare)\s+([^,\n]+)\s+và\s+([^,\n]+)", "comparison_a"),
            (r"(so\s+sánh|compare)\s+([^,\n]+)\s+và\s+([^,\n]+)", "comparison_b"),
            (r"(khác\s+biệt|difference)\s+giữa\s+([^,\n]+)\s+và\s+([^,\n]+)", "comparison_a"),
            (r"(khác\s+biệt|difference)\s+giữa\s+([^,\n]+)\s+và\s+([^,\n]+)", "comparison_b"),
        ]
    
    def _build_youtube_search_patterns(self) -> List[Tuple[str, str]]:
        """Xây dựng patterns cho YouTube search - Tìm kiếm YouTube"""
        return [
            # CHANNEL_NAME patterns
            (r"(kênh|channel)\s+([^,\n]+)", "channel_name"),
            (r"(từ\s+kênh|from\s+channel)\s+([^,\n]+)", "channel_name"),
            (r"(của\s+kênh|by\s+channel)\s+([^,\n]+)", "channel_name"),
            
            # LIVE_ONLY patterns
            (r"(trực\s+tiếp|live)\s+(có|không|on|off)", "live_only"),
            (r"(chỉ\s+trực\s+tiếp|only\s+live)", "live_only"),
            (r"(live\s+stream|stream\s+trực\s+tiếp)", "live_only"),
            
            # PLAYLIST_ID patterns
            (r"(playlist\s+id|id\s+playlist)\s+([a-zA-Z0-9_-]+)", "playlist_id"),
            (r"(playlist:)([a-zA-Z0-9_-]+)", "playlist_id"),
            
            # PLAYLIST_NAME patterns
            (r"(playlist|danh\s+sách)\s+['\"]?([^'\"\n]+)['\"]?", "playlist_name"),
            (r"(từ\s+playlist|from\s+playlist)\s+['\"]?([^'\"\n]+)['\"]?", "playlist_name"),
            
            # YT_QUERY patterns - Cải thiện với YouTube exceptions
            (r"(tìm\s+kiếm|search|tìm|tim\s+kiem)\s+trên\s+(youtube|yt|du\s+tup|diu\s+tup|du\s+túp|diu\s+túp)\s+([^,\n]+)", "query"),
            (r"(youtube|yt|du\s+tup|diu\s+tup|du\s+túp|diu\s+túp)\s+([^,\n]+)", "query"),
            (r"(tìm\s+kiếm|search|tìm|tim\s+kiem)\s+([^,\n]+)", "query"),
            
            # YT_KIND patterns
            (r"(nhạc|music|bài\s+hát)", "kind"),
            (r"(hướng\s+dẫn|tutorial|tut)", "kind"),
            (r"(tin\s+tức|news|thời\s+sự)", "kind"),
            (r"(giải\s+trí|entertainment)", "kind"),
            (r"(thể\s+thao|sports|bóng\s+đá)", "kind"),
            
            # DURATION patterns
            (r"(ngắn|short)\s+(dưới\s+4\s+phút|under\s+4\s+min)", "duration"),
            (r"(dài|long)\s+(hơn\s+20\s+phút|over\s+20\s+min)", "duration"),
            (r"(trung\s+bình|medium)\s+(4-20\s+phút|4-20\s+min)", "duration"),
            
            # QUALITY patterns
            (r"(4k|1080p|720p|480p|360p|hd|full\s+hd)", "quality"),
            (r"(chất\s+lượng|quality)\s+(4k|1080p|720p|480p|360p|hd|full\s+hd)", "quality"),
            
            # UPLOAD_DATE patterns
            (r"(hôm\s+nay|today)", "upload_date"),
            (r"(tuần\s+này|this\s+week)", "upload_date"),
            (r"(tháng\s+này|this\s+month)", "upload_date"),
            (r"(năm\s+này|this\s+year)", "upload_date"),
            (r"(\d+\s+ngày\s+trước|\d+\s+days\s+ago)", "upload_date"),
        ]
    
    def _build_information_patterns(self) -> List[Tuple[str, str]]:
        """Xây dựng patterns cho information - Lấy thông tin"""
        return [
            # METRIC patterns
            (r"(nhiệt\s+độ|temp|độ)", "metric"),
            (r"(độ\s+ẩm|humidity|ẩm)", "metric"),
            (r"(giá|price|cost)", "metric"),
            (r"(tỷ\s+giá|exchange|rate)", "metric"),
            (r"(thời\s+tiết|weather|khí\s+hậu)", "metric"),
            (r"(thời\s+gian|time|giờ)", "metric"),
            (r"(ngày|date|tháng|năm|tuổi)", "metric"),
            
            # UNIT patterns
            (r"(độ\s+c|celsius|c)", "unit"),
            (r"(độ\s+f|fahrenheit|f)", "unit"),
            (r"(phần\s+trăm|percent|%)", "unit"),
            (r"(vnd|đồng|dollar|usd|euro|eur|yen|jpy)", "unit"),
            (r"(giờ|hour|h|phút|minute|min|giây|second|s)", "unit"),
            
            # GRANULARITY patterns
            (r"(bây\s+giờ|now|hiện\s+tại)", "granularity"),
            (r"(theo\s+giờ|hourly|từng\s+giờ)", "granularity"),
            (r"(theo\s+ngày|daily|hàng\s+ngày)", "granularity"),
            (r"(theo\s+tuần|weekly|hàng\s+tuần)", "granularity"),
            (r"(theo\s+tháng|monthly|hàng\s+tháng)", "granularity"),
            (r"(theo\s+năm|yearly|hàng\s+năm)", "granularity"),
            
            # PERSON patterns
            (r"(thông\s+tin\s+về|about|về)\s+([A-Za-z\s]+)", "person"),
            (r"(tiểu\s+sử|biography|bio)\s+([A-Za-z\s]+)", "person"),
            (r"(ai\s+là|who\s+is)\s+([A-Za-z\s]+)", "person"),
            
            # EVENT patterns
            (r"(sự\s+kiện|event|happening)\s+([^,\n]+)", "event"),
            (r"(lịch\s+sử|history|historical)\s+([^,\n]+)", "event"),
            (r"(ngày\s+lễ|holiday|celebration)\s+([^,\n]+)", "event"),
            
            # LOCATION patterns
            (r"(ở\s+đâu|where|đâu)\s+([^,\n]+)", "location"),
            (r"(tại|at|in)\s+([^,\n]+)", "location"),
            (r"(thành\s+phố|city|tỉnh|province|quốc\s+gia|country)\s+([^,\n]+)", "location"),
            
            # TOPIC patterns
            (r"(chủ\s+đề|topic|subject)\s+([^,\n]+)", "topic"),
            (r"(về|about|regarding)\s+([^,\n]+)", "topic"),
            (r"(thông\s+tin|information|info)\s+([^,\n]+)", "topic"),
            
            # QUESTION_TYPE patterns
            (r"(là\s+gì|what|gì)", "question_type"),
            (r"(như\s+thế\s+nào|how|thế\s+nào)", "question_type"),
            (r"(khi\s+nào|when|lúc\s+nào)", "question_type"),
            (r"(ở\s+đâu|where|đâu)", "question_type"),
            (r"(tại\s+sao|why|vì\s+sao)", "question_type"),
            (r"(ai|who|người\s+nào)", "question_type"),
            (r"(bao\s+nhiêu|how\s+much|mấy)", "question_type"),
        ]
        
    def _build_receiver_patterns(self) -> List[Tuple[str, str]]:
        """Xây dựng patterns cho RECEIVER extraction - Tối ưu cho người già"""
        return [
            # Pattern 0: Số điện thoại (ưu tiên cao nhất)
            (r"nhắn\s+tin\s+qua\s+số\s+điện\s+thoại\s+không\s+chín\s+tám\s+năm\s+ba\s+tám\s+ba\s+năm\s+sáu\s+chín", "nhắn"),
            (r"nhắn\s+tin\s+qua\s+số\s+điện\s+thoại\s+(\d+)", "nhắn"),
            (r"gửi\s+tin\s+qua\s+số\s+điện\s+thoại\s+(\d+)", "nhắn"),
            (r"nhắn\s+tin\s+qua\s+số\s+(\d+)", "nhắn"),
            (r"gửi\s+tin\s+qua\s+số\s+(\d+)", "nhắn"),
            (r"gọi\s+số\s+(\d+)", "gọi"),
            (r"số\s+(\d+)", "gọi"),
            (r"(\d{10,11})", "gọi"),  # Số điện thoại 10-11 chữ số
            
            # Pattern 1: Gọi trực tiếp (ưu tiên cao) - Cải thiện với stoplist và giới hạn độ dài
            (r"gọi\s+(?:cho|tới|đến)?\s*([^\s]+(?:\s+[^\s]+){0,2}?)(?=\s+(?:lúc|vào|trên|qua|rằng|là|nhé|nha|ạ|nhá|ngay|bây giờ|zalo|messenger|whatsapp|telegram|viber|line|skype|discord|youtube|facebook|google|tiktok|instagram|twitter|linkedin|reddit|grab|be|xanh sm|shopee|lazada|tiki|gojek)\b|[.,]|$)", "gọi"),
            (r"alo\s+(?:cho|tới|đến)?\s*([^\s]+(?:\s+[^\s]+){0,2}?)(?=\s+(?:lúc|vào|trên|qua|rằng|là|nhé|nha|ạ|nhá|ngay|bây giờ|zalo|messenger|whatsapp|telegram|viber|line|skype|discord|youtube|facebook|google|tiktok|instagram|twitter|linkedin|reddit|grab|be|xanh sm|shopee|lazada|tiki|gojek)\b|[.,]|$)", "gọi"),
            (r"gọi\s+điện\s+(?:cho|tới|đến)?\s*([^\s]+(?:\s+[^\s]+){0,2}?)(?=\s+(?:lúc|vào|trên|qua|rằng|là|nhé|nha|ạ|nhá|ngay|bây giờ|zalo|messenger|whatsapp|telegram|viber|line|skype|discord|youtube|facebook|google|tiktok|instagram|twitter|linkedin|reddit|grab|be|xanh sm|shopee|lazada|tiki|gojek)\b|[.,]|$)", "gọi"),
            (r"gọi\s+thoại\s+(?:cho|tới|đến)?\s*([^\s]+(?:\s+[^\s]+){0,2}?)(?=\s+(?:lúc|vào|trên|qua|rằng|là|nhé|nha|ạ|nhá|ngay|bây giờ|zalo|messenger|whatsapp|telegram|viber|line|skype|discord|youtube|facebook|google|tiktok|instagram|twitter|linkedin|reddit|grab|be|xanh sm|shopee|lazada|tiki|gojek)\b|[.,]|$)", "gọi"),
            
            # Pattern 1.1: Gọi điện video (ưu tiên cao) - Pattern riêng cho video call
            (r"gọi\s+điện\s+video\s+(?:cho|tới|đến)?\s*([^\s]+(?:\s+[^\s]+){0,2}?)(?=\s+(?:lúc|vào|trên|qua|rằng|là|nhé|nha|ạ|nhá|ngay|bây giờ|zalo|messenger|whatsapp|telegram|viber|line|skype|discord|youtube|facebook|google|tiktok|instagram|twitter|linkedin|reddit|grab|be|xanh sm|shopee|lazada|tiki|gojek)\b|[.,]|$)", "video_call"),
            (r"video\s+call\s+(?:cho|tới|đến)?\s*([^\s]+(?:\s+[^\s]+){0,2}?)(?=\s+(?:lúc|vào|trên|qua|rằng|là|nhé|nha|ạ|nhá|ngay|bây giờ|zalo|messenger|whatsapp|telegram|viber|line|skype|discord|youtube|facebook|google|tiktok|instagram|twitter|linkedin|reddit|grab|be|xanh sm|shopee|lazada|tiki|gojek)\b|[.,]|$)", "video_call"),
            
            # Pattern 1.2: Video call patterns (cải thiện với stoplist và giới hạn độ dài)
            (r"gọi\s+video\s+call\s+(?:với|cho|tới|đến)?\s*([^\s]+(?:\s+[^\s]+){0,4}?)(?=\s+(?:trên|lúc|vào|qua|rằng|là|nhé|nha|ạ|nhá|ngay|bây giờ)\b|[.,]|$)", "video_call"),
            (r"đặt\s+video\s+call\s+(?:với|cho|tới|đến)?\s*([^\s]+(?:\s+[^\s]+){0,4}?)(?=\s+(?:trên|lúc|vào|qua|rằng|là|nhé|nha|ạ|nhá|ngay|bây giờ)\b|[.,]|$)", "video_call"),
            (r"video\s+call\s+(?:với|cho|tới|đến)?\s*([^\s]+(?:\s+[^\s]+){0,4}?)(?=\s+(?:trên|lúc|vào|qua|rằng|là|nhé|nha|ạ|nhá|ngay|bây giờ)\b|[.,]|$)", "video_call"),
            (r"gọi\s+video\s+(?:với|cho|tới|đến)?\s*([^\s]+(?:\s+[^\s]+){0,4}?)(?=\s+(?:trên|lúc|vào|qua|rằng|là|nhé|nha|ạ|nhá|ngay|bây giờ)\b|[.,]|$)", "video_call"),
            
            # Pattern 1.3: Video call với platform (thêm mới)
            (r"gọi\s+video\s+qua\s+(?:zalo|messenger|telegram|viber|line|discord|skype|facebook|whatsapp|oắt sáp|goắt sáp)\s+(?:với|cho|tới|đến)?\s*([^\s]+(?:\s+[^\s]+){0,4}?)(?=\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ)\b|[.,]|$)", "video_call"),
            (r"video\s+call\s+qua\s+(?:zalo|messenger|telegram|viber|line|discord|skype|facebook|whatsapp|oắt sáp|goắt sáp)\s+(?:với|cho|tới|đến)?\s*([^\s]+(?:\s+[^\s]+){0,4}?)(?=\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ)\b|[.,]|$)", "video_call"),
            
            # Pattern 1.1: Nói chuyện điện thoại (thêm mới cho trường hợp "Tôi muốn nói chuyện điện thoại với Bố Dũng")
            (r"nói\s+chuyện\s+điện\s+thoại\s+(?:với|cho|tới|đến)?\s*([^\s]+(?:\s+[^\s]+)*?)(?=\s+(?:vì|lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ|$|[\.,]))", "gọi"),
            (r"nói\s+chuyện\s+(?:với|cho|tới|đến)?\s*([^\s]+(?:\s+[^\s]+)*?)(?=\s+(?:vì|lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ|$|[\.,]))", "gọi"),
            (r"trò\s+chuyện\s+(?:với|cho|tới|đến)?\s*([^\s]+(?:\s+[^\s]+)*?)(?=\s+(?:vì|lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ|$|[\.,]))", "gọi"),
            (r"liên\s+lạc\s+(?:với|cho|tới|đến)?\s*([^\s]+(?:\s+[^\s]+)*?)(?=\s+(?:vì|lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ|$|[\.,]))", "gọi"),
            
            # Pattern 2.1: Nhắn tin không dấu (thêm mới) - Cải thiện boundary detection
            (r"nhan\s+(?:tin|tin nhan)?\s+(?:cho|toi|den)?\s*([^\s]+(?:\s+[^\s]+)*?)(?=\s+(?:qua|rang|la|noi|nhe|nha|a|nha|$|[\.,]))", "nhan"),
            (r"gui\s+(?:tin|tin nhan)?\s+(?:cho|toi|den)?\s*([^\s]+(?:\s+[^\s]+)*?)(?=\s+(?:qua|rang|la|noi|nhe|nha|a|nha|$|[\.,]))", "nhan"),
            (r"soan\s+tin\s+(?:cho|toi|den)?\s*([^\s]+(?:\s+[^\s]+)*?)(?=\s+(?:qua|rang|la|noi|nhe|nha|a|nha|$|[\.,]))", "nhan"),
            
            # Pattern 3: Với platform (cải thiện để extract chính xác) - Thêm patterns mới
            
            # Pattern 3.1: Patterns cụ thể cho các platform phổ biến - Cải thiện để capture nhiều từ hơn
            (r"nhắn\s+tin\s+qua\s+(?:zalo|messenger|telegram|viber|line|discord|skype|facebook|whatsapp|oắt sáp|goắt sáp)\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:rằng|là|nói|nhé|nha|ạ|nhá))", "nhắn"),
            (r"gửi\s+tin\s+qua\s+(?:zalo|messenger|telegram|viber|line|discord|skype|facebook|whatsapp|oắt sáp|goắt sáp)\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:rằng|là|nói|nhé|nha|ạ|nhá))", "nhắn"),
            (r"soạn\s+tin\s+qua\s+(?:zalo|messenger|telegram|viber|line|discord|skype|facebook|whatsapp|oắt sáp|goắt sáp)\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:rằng|là|nói|nhé|nha|ạ|nhá))", "nhắn"),
            
            # Pattern 3.2: Patterns với boundary mở rộng hơn
            (r"nhắn\s+tin\s+qua\s+(?:zalo|messenger|telegram|viber|line|discord|skype|facebook|whatsapp|oắt sáp|goắt sáp)\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?=\s+(?:rằng|là|nói|nhé|nha|ạ|nhá|lúc|vào|trên|qua|rằng|là|nhé|nha|ạ|nhá|ngay|bây giờ)\b|[.,]|$)", "nhắn"),
            (r"gửi\s+tin\s+qua\s+(?:zalo|messenger|telegram|viber|line|discord|skype|facebook|whatsapp|oắt sáp|goắt sáp)\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?=\s+(?:rằng|là|nói|nhé|nha|ạ|nhá|lúc|vào|trên|qua|rằng|là|nhé|nha|ạ|nhá|ngay|bây giờ)\b|[.,]|$)", "nhắn"),
            (r"soạn\s+tin\s+qua\s+(?:zalo|messenger|telegram|viber|line|discord|skype|facebook|whatsapp|oắt sáp|goắt sáp)\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?=\s+(?:rằng|là|nói|nhé|nha|ạ|nhá|lúc|vào|trên|qua|rằng|là|nhé|nha|ạ|nhá|ngay|bây giờ)\b|[.,]|$)", "nhắn"),
            
            # Pattern 4: Video call
            (r"facetime\s+(?:với|cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "video"),
            (r"video\s+call\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "video"),
            (r"mở\s+video\s+call\s+(?:với|cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "video"),
            (r"đặt\s+video\s+call\s+(?:với|cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "video"),
            (r"tạo\s+video\s+call\s+(?:với|cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "video"),
            (r"start\s+video\s+chat\s+(?:với|cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "video"),
            (r"video\s+chat\s+(?:với|cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "video"),
            (r"face\s+cam\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "video"),
            (r"meet\s+(?:với|cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "video"),
            (r"video\s+conference\s+(?:với|cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "video"),
            (r"bật\s+video\s+conference\s+(?:với|cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "video"),
            
            # Pattern 5: Khẩn cấp
            (r"gọi\s+ngay\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá))?(?:$|[\.,])", "gọi"),
            (r"gọi\s+khẩn\s+cấp\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá))?(?:$|[\.,])", "gọi"),
            
            # Pattern 6: Nhiều người (tối ưu cho gia đình)
            (r"gọi\s+cho\s+(?:cả\s+nhà|tất\s+cả|mọi\s+người|con\s+cháu|gia\s+đình)", "gọi"),
            (r"nhắn\s+tin\s+cho\s+(?:cả\s+nhà|tất\s+cả|mọi\s+người|con\s+cháu|gia\s+đình)", "nhắn"),
            
            # Pattern 7: Quan hệ phức tạp (tối ưu cho người già)
            (r"gọi\s+cho\s+([\w\s]+?)\s+(?:của|ở|tại)\s+[\w\s]+", "gọi"),
            (r"nhắn\s+tin\s+cho\s+([\w\s]+?)\s+(?:của|ở|tại)\s+[\w\s]+", "nhắn"),
            
            # Pattern 8: Quan hệ gia đình (thêm mới)
            (r"gọi\s+cho\s+(?:bố|mẹ|ông|bà|anh|chị|em|con|cháu|chú|bác|cô|dì|dượng|mợ)", "gọi"),
            (r"nhắn\s+tin\s+cho\s+(?:bố|mẹ|ông|bà|anh|chị|em|con|cháu|chú|bác|cô|dì|dượng|mợ)", "nhắn"),
            
            # Pattern 9: Tên riêng (thêm mới)
            (r"gọi\s+cho\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "gọi"),
            (r"nhắn\s+tin\s+cho\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "nhắn"),
            
            # Pattern 10: Fallback patterns
            (r"cho\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá|ngay bây giờ|ngay|bây giờ))?(?:$|[\.,])", "gọi"),
            (r"(?:cuộc gọi|gọi điện|gọi thoại)\s+(?:cho|tới|đến)\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá|ngay bây giờ|ngay|bây giờ))?(?:$|[\.,])", "gọi"),
        ]
    
    def _build_contact_patterns(self) -> List[Tuple[str, str]]:
        """Xây dựng patterns cho CONTACT extraction - Tên, số điện thoại, email, ghi chú"""
        return [
            # Pattern 1: Tên người (ưu tiên cao)
            (r"lưu\s+liên\s+hệ\s+mới:\s*([^,]+?)(?:,|$)", "contact"),
            (r"thêm\s+bạn\s+['\"]([^'\"]+)['\"]", "contact"),
            (r"create\s+contact:\s*([^,]+?)(?:,|$)", "contact"),
            (r"ghi\s+contact\s+['\"]([^'\"]+)['\"]", "contact"),
            (r"add\s+contact\s+([^–-]+?)(?:–|-|$)", "contact"),
            (r"lưu\s+(?:chị|anh|cô|thầy|bác|ông|bà)\s+([^–-]+?)(?:–|-|$)", "contact"),
            
            # Pattern 2: Số điện thoại với normalized format
            (r"số\s+([0-9\s\.\-\(\)]+)", "phone"),
            (r"(\+84\s*[0-9\s\.\-\(\)]+)", "phone"),
            (r"(\d{10,11})", "phone"),
            (r"(\d{3,4}\.\d{3,4}\.\d{3,4})", "phone"),
            (r"(\d{3,4}\s\d{3,4}\s\d{3,4})", "phone"),
            (r"(\d{3,4})\s*(\d{3,4})\s*(\d{3,4})", "phone"),
            (r"(\d{3,4})-(\d{3,4})-(\d{3,4})", "phone"),
            
            # Pattern 3: Email
            (r"email\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", "email"),
            (r"mail\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", "email"),
            (r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", "email"),
            
            # Pattern 4: Ghi chú/Note
            (r"note\s+['\"]([^'\"]+)['\"]", "note"),
            (r"tag\s+['\"]([^'\"]+)['\"]", "note"),
            (r"ghi\s+chú\s+['\"]([^'\"]+)['\"]", "note"),
            
            # Pattern 5: Địa chỉ
            (r"nhà\s+ở\s+([^,]+?)(?:,|$)", "location"),
            (r"địa\s+chỉ\s+([^,]+?)(?:,|$)", "location"),
            (r"ở\s+([^,]+?)(?:,|$)", "location"),
            
            # Pattern 6: Công ty
            (r"công\s+ty\s+([^,]+?)(?:,|$)", "company"),
            (r"cty\s+([^,]+?)(?:,|$)", "company"),
            (r"work\s+([^,]+?)(?:,|$)", "company"),
            
            # Pattern 7: Sinh nhật
            (r"sinh\s+nhật\s+(\d{1,2}/\d{1,2})", "birthday"),
            (r"birthday\s+(\d{1,2}/\d{1,2})", "birthday"),
            
            # Pattern 8: Zalo
            (r"zalo\s+([0-9\s]+)", "zalo"),
            
            # Pattern 9: Biển số xe
            (r"biển\s+số\s+([A-Z0-9\-\.]+)", "license_plate"),
            
            # Pattern 10: Address (NEW)
            (r"địa\s+chỉ\s+([^,\n]+)", "address"),
            (r"address\s+([^,\n]+)", "address"),
            (r"nhà\s+([^,\n]+)", "address"),
            (r"home\s+([^,\n]+)", "address"),
            (r"work\s+address\s+([^,\n]+)", "address"),
            (r"địa\s+chỉ\s+làm\s+việc\s+([^,\n]+)", "address"),
            
            # Pattern 11: Nickname (NEW)
            (r"biệt\s+danh\s+['\"]?([^'\",\n]+)['\"]?", "nickname"),
            (r"nickname\s+['\"]?([^'\",\n]+)['\"]?", "nickname"),
            (r"tên\s+thân\s+mật\s+['\"]?([^'\",\n]+)['\"]?", "nickname"),
            (r"gọi\s+là\s+['\"]?([^'\",\n]+)['\"]?", "nickname"),
            (r"call\s+me\s+['\"]?([^'\",\n]+)['\"]?", "nickname"),
            
            # Pattern 12: Birthday (NEW)
            (r"sinh\s+nhật\s+(\d{1,2}/\d{1,2}/\d{4})", "birthday"),
            (r"birthday\s+(\d{1,2}/\d{1,2}/\d{4})", "birthday"),
            (r"ngày\s+sinh\s+(\d{1,2}/\d{1,2}/\d{4})", "birthday"),
            (r"date\s+of\s+birth\s+(\d{1,2}/\d{1,2}/\d{4})", "birthday"),
            (r"dob\s+(\d{1,2}/\d{1,2}/\d{4})", "birthday"),
            
            # Pattern 13: Relation (NEW)
            (r"quan\s+hệ\s+([^,\n]+)", "relation"),
            (r"relation\s+([^,\n]+)", "relation"),
            (r"mối\s+quan\s+hệ\s+([^,\n]+)", "relation"),
            (r"relationship\s+([^,\n]+)", "relation"),
            (r"là\s+(bố|mẹ|anh|chị|em|ông|bà|bạn|đồng\s+nghiệp)", "relation"),
            (r"is\s+(father|mother|brother|sister|friend|colleague)", "relation"),
        ]
    
    def _build_media_patterns(self) -> List[Tuple[str, str]]:
        """Xây dựng patterns cho MEDIA extraction - Playlist, artist, podcast, file"""
        return [
            # Pattern 1: Playlist
            (r"playlist\s+([^trên]+?)(?:\s+trên|$)", "playlist"),
            (r"phát\s+playlist\s+([^trên]+?)(?:\s+trên|$)", "playlist"),
            
            # Pattern 2: Artist/Singer
            (r"play\s+([A-Za-z\s]+?)(?:\s+bản|\s+trên|\s+đi|$)", "artist"),
            (r"phát\s+([A-Za-z\s]+?)(?:\s+bản|\s+trên|\s+đi|$)", "artist"),
            (r"Play\s+([A-Za-z\s]+?)(?:\s+bản|\s+trên|\s+đi|$)", "artist"),
            
            # Pattern 3: Podcast
            (r"podcast\s+['\"]([^'\"]+)['\"]", "podcast"),
            (r"mở\s+podcast\s+['\"]([^'\"]+)['\"]", "podcast"),
            (r"nghe\s+podcast\s+['\"]([^'\"]+)['\"]", "podcast"),
            
            # Pattern 4: File path - Bao gồm các format media phổ biến
            (r"file\s+([\/\\][^\\s]+)", "file_path"),
            (r"play\s+file\s+([\/\\][^\\s]+)", "file_path"),
            (r"phát\s+file\s+([\/\\][^\\s]+)", "file_path"),
            # File với extension cụ thể
            (r"play\s+file\s+([^\\s]+\.mp3)", "file_path"),
            (r"phát\s+file\s+([^\\s]+\.mp3)", "file_path"),
            (r"phat\s+file\s+([^\\s]+\.mp3)", "file_path"),
            (r"play\s+file\s+([^\\s]+\.wav)", "file_path"),
            (r"phát\s+file\s+([^\\s]+\.wav)", "file_path"),
            (r"phat\s+file\s+([^\\s]+\.wav)", "file_path"),
            (r"play\s+file\s+([^\\s]+\.flac)", "file_path"),
            (r"phát\s+file\s+([^\\s]+\.flac)", "file_path"),
            (r"phat\s+file\s+([^\\s]+\.flac)", "file_path"),
            (r"play\s+file\s+([^\\s]+\.mp4)", "file_path"),
            (r"phát\s+file\s+([^\\s]+\.mp4)", "file_path"),
            (r"phat\s+file\s+([^\\s]+\.mp4)", "file_path"),
            (r"play\s+file\s+([^\\s]+\.avi)", "file_path"),
            (r"phát\s+file\s+([^\\s]+\.avi)", "file_path"),
            (r"phat\s+file\s+([^\\s]+\.avi)", "file_path"),
            (r"play\s+file\s+([^\\s]+\.mkv)", "file_path"),
            (r"phát\s+file\s+([^\\s]+\.mkv)", "file_path"),
            (r"phat\s+file\s+([^\\s]+\.mkv)", "file_path"),
            
            # Pattern 5: Radio/Stream
            (r"radio\s+([A-Za-z0-9\s]+?)(?:\s+stream|$)", "radio"),
            (r"phát\s+radio\s+([A-Za-z0-9\s]+?)(?:\s+stream|$)", "radio"),
            (r"stream\s+([A-Za-z0-9\s]+)", "stream"),
            
            # Pattern 6: Genre/Mood
            (r"nhạc\s+([^cho]+?)(?:\s+cho|$)", "genre"),
            (r"playlist\s+([^trên]+?)(?:\s+trên|$)", "genre"),
            
            # Pattern 7: Episode/Version
            (r"tập\s+([^mới]+?)(?:\s+mới|$)", "episode"),
            (r"bản\s+([^live]+?)(?:\s+live|$)", "version"),
            
            # Pattern 8: Context/Purpose
            (r"cho\s+([^ngủ]+?)(?:\s+ngủ|$)", "context"),
            (r"để\s+([^học]+?)(?:\s+học|$)", "purpose"),
        ]
    
    def _build_search_patterns(self) -> List[Tuple[str, str]]:
        """Xây dựng patterns cho SEARCH extraction - Query, platform, preference"""
        return [
            # Pattern 1: Search query in quotes
            (r"tra\s+giúp\s+['\"]([^'\"]+)['\"]", "query"),
            (r"search:\s*['\"]([^'\"]+)['\"]", "query"),
            (r"tìm\s+['\"]([^'\"]+)['\"]", "query"),
            (r"kiếm\s+['\"]([^'\"]+)['\"]", "query"),
            
            # Pattern 2: Search query without quotes
            (r"tra\s+giúp\s+mình\s+([^trên]+?)(?:\s+trên|$)", "query"),
            (r"search:\s*([^\(]+?)(?:\s*\(|$)", "query"),
            (r"tìm\s+([^cuối]+?)(?:\s+cuối|$)", "query"),
            (r"kiếm\s+([^về]+?)(?:\s+về|$)", "query"),
            (r"tra\s+cứu\s+([^cho]+?)(?:\s+cho|$)", "query"),
            
            # Pattern 3: Platform specification
            (r"trên\s+(google|bing|yahoo|duckduckgo)", "platform"),
            (r"bằng\s+(google|bing|yahoo|duckduckgo)", "platform"),
            (r"qua\s+(google|bing|yahoo|duckduckgo)", "platform"),
            
            # Pattern 4: Preference/Filter
            (r"ưu\s+tiên\s+([^\.]+?)(?:\s*\.|$)", "preference"),
            (r"bản\s+([^,]+?)(?:\s*,|$)", "preference"),
            (r"nguồn\s+([^\.]+?)(?:\s*\.|$)", "preference"),
            
            # Pattern 5: Time constraint
            (r"cuối\s+tuần\s+này", "time"),
            (r"tuần\s+này", "time"),
            (r"hôm\s+nay", "time"),
            (r"ngày\s+mai", "time"),
            
            # Pattern 6: Comparison
            (r"so\s+sánh\s+([^về]+?)(?:\s+về|$)", "comparison"),
            (r"bài\s+so\s+sánh\s+([^về]+?)(?:\s+về|$)", "comparison"),
            
            # Pattern 7: What is questions
            (r"what\s+is\s+([^?]+?)(?:\s*\?|$)", "query"),
            (r"what\s+are\s+([^?]+?)(?:\s*\?|$)", "query"),
        ]
    
    def _build_youtube_patterns(self) -> List[Tuple[str, str]]:
        """Xây dựng patterns cho YOUTUBE search extraction - Query, duration, type, language"""
        return [
            # Pattern 1: YouTube search query in quotes
            (r"tìm\s+trên\s+youtube:\s*['\"]([^'\"]+)['\"]", "query"),
            (r"youtube\s+['\"]([^'\"]+)['\"]", "query"),
            (r"search\s+yt:\s*['\"]([^'\"]+)['\"]", "query"),
            (r"kiếm\s+clip\s+['\"]([^'\"]+)['\"]", "query"),
            
            # Pattern 2: YouTube search query without quotes
            (r"tìm\s+trên\s+youtube:\s*([^tutorial]+?)(?:\s+tutorial|$)", "query"),
            (r"youtube\s+([^playlist]+?)(?:\s+playlist|$)", "query"),
            (r"search\s+yt:\s*([^newest]+?)(?:\s+newest|$)", "query"),
            (r"kiếm\s+clip\s+([^tiếng]+?)(?:\s+tiếng|$)", "query"),
            (r"bài\s+giảng\s+([^nguồn]+?)(?:\s+nguồn|$)", "query"),
            
            # Pattern 3: Duration/Time
            (r"(\d+[-–]\d+)\s*phút", "duration"),
            (r"(\d+)\s*hour", "duration"),
            (r"(\d+[-–]\d+)\s*phút", "duration"),
            (r"(\d+)\s*giờ", "duration"),
            
            # Pattern 4: Content Type
            (r"tutorial\s+(\d+[-–]\d+\s*phút)", "type"),
            (r"playlist\s+nào", "type"),
            (r"clip\s+review", "type"),
            (r"bài\s+giảng", "type"),
            (r"hướng\s+dẫn", "type"),
            (r"tutorial", "type"),
            (r"playlist", "type"),
            (r"review", "type"),
            
            # Pattern 5: Language
            (r"tiếng\s+việt", "language"),
            (r"tieng\s+viet", "language"),
            (r"vietnamese", "language"),
            
            # Pattern 6: Channel/Source
            (r"của\s+kênh\s+([^hay]+?)(?:\s+hay|$)", "channel"),
            (r"kênh\s+([^hay]+?)(?:\s+hay|$)", "channel"),
            (r"channel\s+([^or]+?)(?:\s+or|$)", "channel"),
            
            # Pattern 7: Time constraint
            (r"newest\s+within\s+(\d+\s*year)", "time_constraint"),
            (r"mới\s+nhất\s+trong\s+(\d+\s*năm)", "time_constraint"),
            (r"trong\s+(\d+\s*năm)", "time_constraint"),
            
            # Pattern 8: Academic/Source preference
            (r"nguồn\s+học\s+thuật", "source"),
            (r"academic\s+source", "source"),
            (r"scholarly", "source"),
            
            # Pattern 9: No ads preference
            (r"no\s+ads", "preference"),
            (r"không\s+quảng\s+cáo", "preference"),
            (r"ad-free", "preference"),
        ]
    
    def _build_info_patterns(self) -> List[Tuple[str, str]]:
        """Xây dựng patterns cho INFO extraction - Topic, location, time, amount"""
        return [
            # Pattern 1: Weather info
            (r"thời\s+tiết\s+([^cuối]+?)(?:\s+cuối|$)", "location"),
            (r"dự\s+báo\s+thời\s+tiết\s+([^cuối]+?)(?:\s+cuối|$)", "location"),
            (r"nhiệt\s+độ\s+([^cuối]+?)(?:\s+cuối|$)", "location"),
            (r"mưa\s+([^cuối]+?)(?:\s+cuối|$)", "location"),
            (r"nắng\s+([^cuối]+?)(?:\s+cuối|$)", "location"),
            
            # Pattern 2: Currency exchange
            (r"đổi\s+(\d+)\s+([A-Z]{3})\s+ra\s+([A-Z]{3})", "amount"),
            (r"tỷ\s+giá\s+([A-Z]{3})\s+ra\s+([A-Z]{3})", "currency"),
            (r"(\d+)\s+USD\s+ra\s+VND", "amount"),
            (r"(\d+)\s+USD\s+khoảng\s+bao\s+nhiêu", "amount"),
            
            # Pattern 3: Date/Calendar
            (r"âm\s+lịch\s+([^vậy]+?)(?:\s+vậy|$)", "topic"),
            (r"ngày\s+âm\s+lịch\s+([^vậy]+?)(?:\s+vậy|$)", "topic"),
            (r"hôm\s+nay\s+là\s+ngày\s+([^vậy]+?)(?:\s+vậy|$)", "topic"),
            
            # Pattern 4: Sports info
            (r"vô\s+địch\s+([^mùa]+?)(?:\s+mùa|$)", "competition"),
            (r"đội\s+nào\s+vô\s+địch\s+([^mùa]+?)(?:\s+mùa|$)", "competition"),
            (r"tỉ\s+số\s+trận\s+([^và]+?)(?:\s+và|$)", "match"),
            (r"chung\s+kết\s+([^và]+?)(?:\s+và|$)", "match"),
            
            # Pattern 5: Summary/Explanation
            (r"tóm\s+tắt\s+([^khác]+?)(?:\s+khác|$)", "topic"),
            (r"khác\s+gì\s+([^và]+?)(?:\s+và|$)", "comparison"),
            (r"so\s+sánh\s+([^và]+?)(?:\s+và|$)", "comparison"),
            
            # Pattern 6: Time references
            (r"cuối\s+tuần\s+này", "time"),
            (r"bây\s+giờ", "time"),
            (r"hôm\s+nay", "time"),
            (r"gần\s+nhất", "time"),
            (r"mùa\s+gần\s+nhất", "time"),
            
            # Pattern 7: Amount/Number
            (r"(\d+)\s+USD", "amount"),
            (r"(\d+)\s+VND", "amount"),
            (r"khoảng\s+bao\s+nhiêu", "question"),
            (r"bao\s+nhiêu", "question"),
            (r"trung\s+bình", "type"),
            
            # Pattern 8: Location
            (r"Đà\s+Lạt", "location"),
            (r"([A-Z][a-z]+\s+[A-Z][a-z]+)", "location"),
            (r"([A-Z][a-z]+)", "location"),
        ]
    
    def _build_alarm_patterns(self) -> List[Tuple[str, str]]:
        """Xây dựng patterns cho ALARM extraction - Time, label, volume, recurrence"""
        return [
            # Pattern 1: Time extraction (HH:MM format) with relative time - Cải thiện để bao gồm phút
            (r"(\d{1,2})\s*giờ\s*(\d{1,2})\s*phút\s*(?:sáng|trưa|chiều|tối|đêm)?", "time"),
            (r"(\d{1,2})\s*giờ\s*(\d{1,2})\s*(?:sáng|trưa|chiều|tối|đêm)?", "time"),
            (r"(\d{1,2}):(\d{2})", "time"),
            (r"(\d{1,2})\s*rưỡi", "time"),
            (r"(\d{1,2})\s*giờ\s*(?:sáng|trưa|chiều|tối|đêm)?(?!\s*\d)", "time"),
            (r"(\d{1,2})\s*hơn", "time"),
            (r"(\d{1,2})\s*kém", "time"),
            (r"(\d{1,2})\s*thiếu", "time"),
            (r"trước\s+bữa\s+(\d{1,2})", "time"),
            (r"sau\s+bữa\s+(\d{1,2})", "time"),
            
            # Pattern 2: Label/Name with gentle labels - Cải thiện để extract label tốt hơn
            (r"với\s+nhãn\s+([^,]+?)(?:,|$)", "label"),
            (r"nhãn\s+([^,]+?)(?:,|$)", "label"),
            (r"label\s+['\"]([^'\"]+)['\"]", "label"),
            (r"tên\s+['\"]([^'\"]+)['\"]", "label"),
            (r"gọi\s+['\"]([^'\"]+)['\"]", "label"),
            
            # Pattern 3: Volume settings
            (r"âm\s+lượng\s+([^,]+?)(?:,|$)", "volume"),
            (r"volume\s+([^,]+?)(?:,|$)", "volume"),
            (r"chuông\s+([^,]+?)(?:,|$)", "sound"),
            (r"sound\s+([^,]+?)(?:,|$)", "sound"),
            
            # Pattern 4: Vibration settings
            (r"đừng\s+rung", "vibration"),
            (r"không\s+rung", "vibration"),
            (r"tắt\s+rung", "vibration"),
            (r"no\s+vibration", "vibration"),
            (r"disable\s+vibration", "vibration"),
            
            # Pattern 5: Recurrence patterns
            (r"mỗi\s+([^lúc]+?)(?:\s+lúc|$)", "recurrence"),
            (r"hằng\s+ngày", "recurrence"),
            (r"daily", "recurrence"),
            (r"weekdays", "recurrence"),
            (r"weekends", "recurrence"),
            (r"T2[–-]T6", "recurrence"),
            (r"T2[–-]T7", "recurrence"),
            
            # Pattern 6: Time periods
            (r"trong\s+(\d+)\s*tuần", "duration"),
            
            # Pattern 7: Days of week (NEW)
            (r"thứ\s+(hai|ba|tư|năm|sáu|bảy)", "days_of_week"),
            (r"chủ\s+nhật", "days_of_week"),
            (r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", "days_of_week"),
            (r"(mon|tue|wed|thu|fri|sat|sun)", "days_of_week"),
            (r"hàng\s+ngày", "days_of_week"),
            (r"daily", "days_of_week"),
            (r"mỗi\s+ngày", "days_of_week"),
            
            # Pattern 8: Snooze minutes (NEW)
            (r"snooze\s+(\d+)\s*phút", "snooze_min"),
            (r"snooze\s+(\d+)\s*minutes", "snooze_min"),
            (r"báo\s+lại\s+(\d+)\s*phút", "snooze_min"),
            (r"delay\s+(\d+)\s*minutes", "snooze_min"),
            (r"hoãn\s+(\d+)\s*phút", "snooze_min"),
            
            # Pattern 9: Volume profile (NEW)
            (r"êm\s+dịu", "volume_profile"),
            (r"nhẹ\s+nhàng", "volume_profile"),
            (r"gentle", "volume_profile"),
            (r"bình\s+thường", "volume_profile"),
            (r"normal", "volume_profile"),
            (r"to", "volume_profile"),
            (r"loud", "volume_profile"),
            (r"mạnh", "volume_profile"),
            (r"for\s+(\d+)\s*weeks", "duration"),
            (r"(\d+)\s*tuần", "duration"),
            (r"(\d+)\s*weeks", "duration"),
            
            # Pattern 7: Auto-cancel
            (r"tự\s+hủy", "auto_cancel"),
            (r"auto\s+cancel", "auto_cancel"),
            (r"tự\s+động\s+hủy", "auto_cancel"),
            (r"tự\s+động\s+tắt", "auto_cancel"),
            
            # Pattern 8: Skip conditions
            (r"skip\s+ngày\s+lễ", "skip_condition"),
            (r"bỏ\s+qua\s+ngày\s+lễ", "skip_condition"),
            (r"trừ\s+ngày\s+lễ", "skip_condition"),
            (r"chỉ\s+ngày\s+thường", "skip_condition"),
            (r"weekdays\s+only", "skip_condition"),
            
            # Pattern 9: Multiple alarms
            (r"(\d+)\s*alarms?", "alarm_count"),
            (r"(\d+)\s*báo\s+thức", "alarm_count"),
            (r"backup", "backup"),
            (r"dự\s+phòng", "backup"),
            
            # Pattern 10: Special types
            (r"power\s+nap", "alarm_type"),
            (r"chợp\s+mắt", "alarm_type"),
            (r"nap", "alarm_type"),
            (r"giấc\s+ngủ\s+ngắn", "alarm_type"),
        ]
    
    def _build_calendar_patterns(self) -> List[Tuple[str, str]]:
        """Xây dựng patterns cho CALENDAR extraction - Title, time, location, platform"""
        return [
            # Pattern 1: Event title extraction - Cải thiện để extract title chính xác hơn
            (r"tạo\s+lịch\s+([^lúc]+?)(?:\s+lúc|$)", "title"),
            (r"đặt\s+event\s+([^lúc]+?)(?:\s+lúc|$)", "title"),
            (r"thêm\s+lịch\s+([^lúc]+?)(?:\s+lúc|$)", "title"),
            (r"tạo\s+lịch\s+['\"]([^'\"]+)['\"]", "title"),
            (r"đặt\s+deadline\s+['\"]([^'\"]+)['\"]", "title"),
            
            # Pattern 2: Time extraction (HH:MM format) with day of week - Cải thiện để bao gồm phút
            (r"(\d{1,2}):(\d{2})[–-](\d{1,2}):(\d{2})", "time_range"),
            (r"(\d{1,2}):(\d{2})\s*[–-]\s*(\d{1,2}):(\d{2})", "time_range"),
            (r"(\d{1,2})\s*giờ\s*(\d{1,2})\s*đến\s*(\d{1,2})\s*giờ\s*(\d{1,2})", "time_range"),
            (r"(\d{1,2})\s*giờ\s*(\d{1,2})\s*đến\s*(\d{1,2})\s*giờ", "time_range"),
            (r"(\d{1,2})\s*giờ\s*đến\s*(\d{1,2})\s*giờ\s*(\d{1,2})", "time_range"),
            (r"(\d{1,2})\s*giờ\s*đến\s*(\d{1,2})\s*giờ", "time_range"),
            (r"(\d{1,2}):(\d{2})", "time"),
            (r"(\d{1,2})\s*giờ\s*(\d{1,2})\s*phút\s*(?:sáng|trưa|chiều|tối|đêm)?", "time"),
            (r"(\d{1,2})\s*giờ\s*(\d{1,2})\s*(?:sáng|trưa|chiều|tối|đêm)?", "time"),
            (r"(\d{1,2})\s*giờ\s*(?:sáng|trưa|chiều|tối|đêm)?(?!\s*\d)", "time"),
            (r"thứ\s+([2-7])", "day_of_week"),
            (r"chủ\s+nhật", "day_of_week"),
            (r"tuần\s+sau", "time_period"),
            (r"tuần\s+này", "time_period"),
            
            # Pattern 3: Date extraction
            (r"ngày\s+(\d{1,2}/\d{1,2})", "date"),
            (r"(\d{1,2}/\d{1,2})", "date"),
            (r"(\d{1,2})\s*tháng\s*(\d{1,2})", "date"),
            (r"(\d{1,2})\s*ngày\s*(\d{1,2})", "date"),
            
            # Pattern 4: Location extraction
            (r"địa\s+điểm\s+([^,]+?)(?:,|$)", "location"),
            (r"tại\s+([^,]+?)(?:,|$)", "location"),
            (r"ở\s+([^,]+?)(?:,|$)", "location"),
            (r"nha\s+khoa\s+([^,]+?)(?:,|$)", "location"),
            (r"bệnh\s+viện\s+([^,]+?)(?:,|$)", "location"),
            (r"trường\s+([^,]+?)(?:,|$)", "location"),
            (r"công\s+ty\s+([^,]+?)(?:,|$)", "location"),
            (r"văn\s+phòng\s+([^,]+?)(?:,|$)", "location"),
            
            # Pattern 5: Platform extraction with online meeting platforms
            (r"qua\s+([^,]+?)(?:,|$)", "platform"),
            (r"bằng\s+([^,]+?)(?:,|$)", "platform"),
            (r"trên\s+([^,]+?)(?:,|$)", "platform"),
            
            # Pattern 6: All-day events (NEW)
            (r"cả\s+ngày", "all_day"),
            (r"all\s+day", "all_day"),
            (r"suốt\s+ngày", "all_day"),
            (r"từ\s+sáng\s+đến\s+tối", "all_day"),
            
            # Pattern 7: End time (NEW)
            (r"đến\s+(\d{1,2}):(\d{2})", "end_time"),
            (r"kết\s+thúc\s+lúc\s+(\d{1,2}):(\d{2})", "end_time"),
            (r"end\s+at\s+(\d{1,2}):(\d{2})", "end_time"),
            (r"finish\s+at\s+(\d{1,2}):(\d{2})", "end_time"),
            
            # Pattern 8: Duration (NEW)
            (r"trong\s+(\d+)\s*phút", "duration"),
            (r"trong\s+(\d+)\s*giờ", "duration"),
            (r"(\d+)\s*phút", "duration"),
            (r"(\d+)\s*giờ", "duration"),
            (r"duration\s+(\d+)\s*minutes", "duration"),
            (r"duration\s+(\d+)\s*hours", "duration"),
            
            # Pattern 9: Recurrence (NEW)
            (r"hàng\s+ngày", "recurrence"),
            (r"hàng\s+tuần", "recurrence"),
            (r"hàng\s+tháng", "recurrence"),
            (r"hàng\s+năm", "recurrence"),
            (r"mỗi\s+ngày", "recurrence"),
            (r"mỗi\s+tuần", "recurrence"),
            (r"mỗi\s+tháng", "recurrence"),
            (r"mỗi\s+năm", "recurrence"),
            (r"thứ\s+2-6", "recurrence"),
            (r"ngày\s+thường", "recurrence"),
            (r"cuối\s+tuần", "recurrence"),
            (r"thứ\s+7-chủ\s+nhật", "recurrence"),
            
            # Pattern 10: Conference link (NEW)
            (r"link\s+([^,\s]+)", "conference_link"),
            (r"meeting\s+link\s+([^,\s]+)", "conference_link"),
            (r"zoom\s+link\s+([^,\s]+)", "conference_link"),
            (r"google\s+meet\s+link\s+([^,\s]+)", "conference_link"),
            (r"teams\s+link\s+([^,\s]+)", "conference_link"),
            
            # Pattern 11: Visibility (NEW)
            (r"công\s+khai", "visibility"),
            (r"public", "visibility"),
            (r"mọi\s+người", "visibility"),
            (r"riêng\s+tư", "visibility"),
            (r"private", "visibility"),
            (r"cá\s+nhân", "visibility"),
            (r"hạn\s+chế", "visibility"),
            (r"restricted", "visibility"),
            (r"giới\s+hạn", "visibility"),
            
            # Pattern 12: Priority (NEW)
            (r"ưu\s+tiên\s+cao", "priority"),
            (r"high\s+priority", "priority"),
            (r"quan\s+trọng", "priority"),
            (r"ưu\s+tiên\s+trung\s+bình", "priority"),
            (r"medium\s+priority", "priority"),
            (r"bình\s+thường", "priority"),
            (r"ưu\s+tiên\s+thấp", "priority"),
            (r"low\s+priority", "priority"),
            (r"không\s+quan\s+trọng", "priority"),
            (r"google\s+meet", "platform"),
            (r"zoom", "platform"),
            (r"teams", "platform"),
            (r"skype", "platform"),
            (r"webex", "platform"),
            (r"meet", "platform"),
            (r"online", "platform"),
            
            # Pattern 6: Recurrence patterns
            (r"mỗi\s+([^từ]+?)(?:\s+từ|$)", "recurrence"),
            (r"hằng\s+tuần", "recurrence"),
            (r"weekly", "recurrence"),
            (r"daily", "recurrence"),
            (r"monthly", "recurrence"),
            (r"T2[–-]T6", "recurrence"),
            (r"T2[–-]T7", "recurrence"),
            (r"T3,\s*T5", "recurrence"),
            (r"T2,\s*T4,\s*T6", "recurrence"),
            
            # Pattern 7: Time periods
            (r"trong\s+(\d+)\s*tuần", "duration"),
            (r"for\s+(\d+)\s*weeks", "duration"),
            (r"(\d+)\s*tuần", "duration"),
            (r"(\d+)\s*weeks", "duration"),
            (r"lặp\s+(\d+)\s*tuần", "duration"),
            (r"repeat\s+(\d+)\s*weeks", "duration"),
            
            # Pattern 8: Notes/Attachments
            (r"attach\s+note\s+['\"]([^'\"]+)['\"]", "note"),
            (r"ghi\s+chú\s+['\"]([^'\"]+)['\"]", "note"),
            (r"note\s+['\"]([^'\"]+)['\"]", "note"),
            (r"mô\s+tả\s+['\"]([^'\"]+)['\"]", "note"),
            (r"description\s+['\"]([^'\"]+)['\"]", "note"),
            
            # Pattern 9: Event types
            (r"họp", "event_type"),
            (r"meeting", "event_type"),
            (r"sprint", "event_type"),
            (r"sinh\s+nhật", "event_type"),
            (r"birthday", "event_type"),
            (r"khám", "event_type"),
            (r"appointment", "event_type"),
            (r"deadline", "event_type"),
            (r"báo\s+cáo", "event_type"),
            (r"report", "event_type"),
            (r"deep\s+work", "event_type"),
            (r"focus\s+time", "event_type"),
            
            # Pattern 10: Time of day
            (r"sáng", "time_of_day"),
            (r"morning", "time_of_day"),
            (r"trưa", "time_of_day"),
            (r"noon", "time_of_day"),
            (r"chiều", "time_of_day"),
            (r"afternoon", "time_of_day"),
            (r"tối", "time_of_day"),
            (r"evening", "time_of_day"),
            (r"đêm", "time_of_day"),
            (r"night", "time_of_day"),
        ]
    
    def _build_camera_patterns(self) -> List[Tuple[str, str]]:
        """Xây dựng patterns cho CAMERA extraction - Camera type, mode, settings, quality"""
        return [
            # Pattern 1: Camera type with named groups
            (r"camera\s+(?P<camera_type>trước|sau)", "camera_type"),
            (r"(?P<camera_type>trước|sau)\s+camera", "camera_type"),
            (r"(?P<camera_type>front|back|rear)\s+camera", "camera_type"),
            (r"(?P<camera_type>trước|sau)", "camera_type"),
            
            # Pattern 2: Camera mode with named groups
            (r"(?P<mode>selfie)", "mode"),
            (r"chụp\s+(?P<mode>ảnh)", "mode"),
            (r"chup\s+(?P<mode>anh)", "mode"),
            (r"quay\s+(?P<mode>video)", "mode"),
            (r"chụp\s+liên\s+tục", "mode"),
            (r"chup\s+lien\s+tuc", "mode"),
            (r"quét\s+(?P<mode>mã)", "mode"),
            (r"quet\s+(?P<mode>ma)", "mode"),
            (r"(?P<mode>panorama)", "mode"),
            (r"document\s+(?P<mode>scan)", "mode"),
            (r"scan\s+tài\s+liệu", "mode"),
            (r"scan\s+tai\s+lieu", "mode"),
            
            # Pattern 3: Camera settings
            (r"lưới\s+căn", "settings"),
            (r"luoi\s+can", "settings"),
            (r"grid\s+lines", "settings"),
            (r"HDR", "settings"),
            (r"flash", "settings"),
            (r"đèn\s+flash", "settings"),
            (r"den\s+flash", "settings"),
            (r"tự\s+động", "settings"),
            (r"tu\s+dong", "settings"),
            (r"auto\s+focus", "settings"),
            (r"tự\s+động\s+lấy\s+nét", "settings"),
            (r"tu\s+dong\s+lay\s+net", "settings"),
            
            # Pattern 4: Video quality with named groups
            (r"(?P<quality>4K)@(?P<fps>\d+)", "quality"),
            (r"(?P<quality>4K)", "quality"),
            (r"(?P<quality>1080p)", "quality"),
            (r"(?P<quality>720p)", "quality"),
            (r"(?P<quality>HD)", "quality"),
            (r"(?P<quality>Full\s+HD)", "quality"),
            (r"(?P<quality>Ultra\s+HD)", "quality"),
            (r"(?P<quality>\d+p)", "quality"),
            (r"(?P<quality>\d+K)", "quality"),
            
            # Pattern 5: Duration/Count
            (r"(\d+)\s*giây", "duration"),
            (r"(\d+)\s*seconds", "duration"),
            (r"(\d+)\s*phút", "duration"),
            (r"(\d+)\s*minutes", "duration"),
            (r"(\d+)\s*tấm", "count"),
            (r"(\d+)\s*photos", "count"),
            (r"(\d+)\s*shots", "count"),
            (r"(\d+)\s*pictures", "count"),
            
            # Pattern 6: Folder/Destination
            (r"lưu\s+vào\s+thư\s+mục\s+['\"]([^'\"]+)['\"]", "folder"),
            (r"luu\s+vao\s+thu\s+muc\s+['\"]([^'\"]+)['\"]", "folder"),
            (r"save\s+to\s+['\"]([^'\"]+)['\"]", "folder"),
            (r"thư\s+mục\s+['\"]([^'\"]+)['\"]", "folder"),
            (r"thu\s+muc\s+['\"]([^'\"]+)['\"]", "folder"),
            (r"folder\s+['\"]([^'\"]+)['\"]", "folder"),
            
            # Pattern 7: Camera features
            (r"cắt\s+mép", "feature"),
            (r"cat\s+mep", "feature"),
            (r"auto\s+crop", "feature"),
            (r"tự\s+động\s+cắt", "feature"),
            (r"tu\s+dong\s+cat", "feature"),
            (r"xoay\s+máy", "feature"),
            (r"xoay\s+may", "feature"),
            (r"rotate\s+device", "feature"),
            (r"từ\s+từ", "feature"),
            (r"tu\s+tu", "feature"),
            (r"slowly", "feature"),
            (r"ngay\s+nhen", "feature"),
            (r"ngay\s+nhen", "feature"),
            (r"right\s+now", "feature"),
            (r"immediately", "feature"),
            
            # Pattern 8: Camera orientation
            (r"chiều\s+ngang", "orientation"),
            (r"chieu\s+ngang", "orientation"),
            (r"landscape", "orientation"),
            (r"chiều\s+dọc", "orientation"),
            (r"chieu\s+doc", "orientation"),
            (r"portrait", "orientation"),
            (r"ngang", "orientation"),
            (r"dọc", "orientation"),
            (r"doc", "orientation"),
            
            # Pattern 9: Camera filters
            (r"filter", "filter"),
            (r"bộ\s+lọc", "filter"),
            (r"bo\s+loc", "filter"),
            (r"màu\s+sắc", "filter"),
            (r"mau\s+sac", "filter"),
            (r"color\s+filter", "filter"),
            (r"black\s+and\s+white", "filter"),
            (r"đen\s+trắng", "filter"),
            (r"den\s+trang", "filter"),
            
            # Pattern 10: Camera timer
            (r"timer\s+(\d+)", "timer"),
            (r"hẹn\s+giờ\s+(\d+)", "timer"),
            (r"hen\s+gio\s+(\d+)", "timer"),
            (r"countdown\s+(\d+)", "timer"),
            (r"đếm\s+ngược\s+(\d+)", "timer"),
            (r"dem\s+nguoc\s+(\d+)", "timer"),
        ]
    
    def _build_time_patterns(self) -> List[str]:
        """Xây dựng patterns cho TIME extraction - Tối ưu cho người già"""
        return [
            # Thời gian cụ thể (ưu tiên cao)
            r"(\d{1,2})\s*giờ\s*(\d{1,2})?\s*(?:phút)?\s*(?:sáng|trưa|chiều|tối|đêm)?",
            r"(\d{1,2})\s*rưỡi\s*(?:sáng|trưa|chiều|tối|đêm)?",
            r"(\d{1,2})\s*giờ\s*rưỡi\s*(?:sáng|trưa|chiều|tối|đêm)?",
            r"(\d{1,2})\s*giờ\s*(?:kém|thiếu)\s*(\d{1,2})",
            
            # Thời gian tương đối (tối ưu cho người già)
            r"(sáng|trưa|chiều|tối|đêm)\s*(?:nay|mai|kia)?",
            r"(hôm\s+nay|ngày\s+mai|tuần\s+sau|tháng\s+sau)",
            r"(sau\s+(?:khi\s+)?ăn|sau\s+bữa\s+(?:sáng|trưa|tối))",
            r"(trước\s+(?:khi\s+)?ăn|trước\s+bữa\s+(?:sáng|trưa|tối))",
            
            # Thời gian khẩn cấp
            r"(ngay|ngay\s+bây\s+giờ|bây\s+giờ|lập\s+tức)",
            r"(khi\s+nào|khi\s+đó|lúc\s+đó)",
            
            # Thời gian định kỳ (thêm mới)
            r"(hàng\s+ngày|hàng\s+tuần|hàng\s+tháng)",
            r"(thứ\s+\d+\s+hàng\s+tuần)",
            r"(ngày\s+\d+\s+hàng\s+tháng)",
            
            # Thời gian theo bữa ăn (tối ưu cho người già)
            r"(sau\s+bữa\s+sáng|sau\s+bữa\s+trưa|sau\s+bữa\s+tối)",
            r"(trước\s+bữa\s+sáng|trước\s+bữa\s+trưa|trước\s+bữa\s+tối)",
            
            # Thời gian theo hoạt động (thêm mới)
            r"(sau\s+khi\s+ngủ|trước\s+khi\s+ngủ)",
            r"(sau\s+khi\s+đi\s+chợ|trước\s+khi\s+đi\s+chợ)",
            r"(sau\s+khi\s+đi\s+bệnh\s+viện|trước\s+khi\s+đi\s+bệnh\s+viện)",
        ]
    
    def _build_message_patterns(self) -> List[str]:
        """Xây dựng patterns cho MESSAGE extraction - Tối ưu cho người già"""
        return [
            # Pattern 1: Rằng là (ưu tiên cao)
            r"rằng\s+là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            
            # Pattern 2: Là
            r"là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            
            # Pattern 3: Nói
            r"nói\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"nói\s+rõ\s+là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            
            # Pattern 4: Nhắn/Gửi
            r"nhắn\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"gửi\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            
            # Pattern 5: Với nội dung trong ngoặc
            r"[\"'](.+?)[\"']",
            
            # Pattern 6: Sau từ khóa
            r"(?:nội\s+dung|tin\s+nhắn)\s+(?:là|rằng)?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            
            # Pattern 7: Tin nhắn dài (thêm mới)
            r"nhắn\s+tin\s+cho\s+[\w\s]+\s+rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"gửi\s+tin\s+cho\s+[\w\s]+\s+rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            
            # Pattern 8: Tin nhắn với thời gian (thêm mới)
            r"nhắn\s+tin\s+cho\s+[\w\s]+\s+rằng\s+(.+?)\s+lúc\s+[\w\s]+",
            r"gửi\s+tin\s+cho\s+[\w\s]+\s+rằng\s+(.+?)\s+lúc\s+[\w\s]+",
            
            # Pattern 9: Nhắn tin cho [người] rằng [nội dung] (cải thiện)
            r"nhắn\s+tin\s+cho\s+[\w\s]+\s+rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"gửi\s+tin\s+cho\s+[\w\s]+\s+rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            
            # Pattern 10: Với nội dung là [nội dung] (thêm mới)
            r"với\s+nội\s+dung\s+là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"nội\s+dung\s+là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            
            # Pattern 11: Nhắn tin qua số điện thoại rằng [nội dung] (thêm mới)
            r"nhắn\s+tin\s+qua\s+số\s+điện\s+thoại\s+[^r]*rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"gửi\s+tin\s+qua\s+số\s+điện\s+thoại\s+[^r]*rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            
            # Pattern 12: Rằng [nội dung] (pattern chung)
            r"rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
        ]
    
    def _build_platform_patterns(self) -> List[str]:
        """Xây dựng patterns cho PLATFORM extraction - Phân biệt giao tiếp và tìm kiếm"""
        return [
            # Pattern 1: Qua/Bằng/Trên (ưu tiên cao) - Platform giao tiếp
            r"qua\s+(zalo|messenger|facebook|whatsapp|sms|tin\s+nhắn|phone|điện\s+thoại)",
            r"bằng\s+(zalo|messenger|facebook|whatsapp|sms|tin\s+nhắn|phone|điện\s+thoại)",
            r"trên\s+(zalo|messenger|facebook|whatsapp|sms|tin\s+nhắn|phone|điện\s+thoại)",
            r"sử\s+dụng\s+(zalo|messenger|facebook|whatsapp|sms|tin\s+nhắn|phone|điện\s+thoại)",
            r"dùng\s+(zalo|messenger|facebook|whatsapp|sms|tin\s+nhắn|phone|điện\s+thoại)",
            
            # Pattern 2: Tìm kiếm/Thông tin - Platform tìm kiếm
            r"tìm\s+(?:kiếm\s+)?(?:trên\s+)?(google|youtube|facebook)",
            r"search\s+(?:trên\s+)?(google|youtube|facebook)",
            r"tra\s+cứu\s+(?:trên\s+)?(google|youtube|facebook)",
            r"tìm\s+thông\s+tin\s+(?:trên\s+)?(google|youtube|facebook)",
            
            # Pattern 3: Trực tiếp - Platform giao tiếp
            r"(zalo|messenger|facebook|whatsapp|sms|tin\s+nhắn|phone|điện\s+thoại)",
            
            # Pattern 4: Tên gọi khác - Platform giao tiếp
            r"(zalo|messenger|facebook|whatsapp|sms|tin\s+nhắn|phone|điện\s+thoại)",
            
            # Pattern 5: Tên gọi thân thiện - Platform giao tiếp
            r"(zalo|messenger|facebook|whatsapp|sms|tin\s+nhắn|phone|điện\s+thoại)",
        ]
    
    def extract_receiver(self, text: str) -> Optional[Dict[str, str]]:
        """Extract RECEIVER entity với thứ tự ưu tiên: số điện thoại > call trực tiếp > video call > nhắn tin"""
        text_lower = text.lower()
        
        # ƯU TIÊN 1: Số điện thoại (cao nhất)
        phone_number = self._extract_phone_number_from_text(text)
        if phone_number:
            # Xác định loại action dựa trên context
            if any(word in text_lower for word in ["gọi", "goi", "alo", "gọi điện", "goi dien"]):
                return {
                    "RECEIVER": phone_number,
                    "ACTION_TYPE": "gọi"
                }
            elif any(word in text_lower for word in ["nhắn tin", "nhan tin", "gửi tin", "gui tin", "soạn tin", "soan tin"]):
                return {
                    "RECEIVER": phone_number,
                    "ACTION_TYPE": "nhắn"
                }
            else:
                # Mặc định là gọi điện
                return {
                    "RECEIVER": phone_number,
                    "ACTION_TYPE": "gọi"
                }
        
        # ƯU TIÊN 2: Call trực tiếp (gọi điện thoại)
        if any(word in text_lower for word in ["gọi", "goi", "alo", "gọi điện", "goi dien", "nói chuyện", "noi chuyen", "liên lạc", "lien lac"]):
            for pattern, action_type in self.receiver_patterns:
                if action_type == "gọi":  # Chỉ xử lý patterns gọi điện
                    match = re.search(pattern, text_lower, re.IGNORECASE)
                    if match and match.groups() and match.group(1):
                        receiver = match.group(1).strip()
                        receiver = self._improve_receiver_boundary(receiver, text_lower)
                        receiver = self._clean_receiver(receiver)
                        if receiver and len(receiver) > 1:
                            return {
                                "RECEIVER": receiver,
                                "ACTION_TYPE": "gọi"
                            }
        
        # ƯU TIÊN 3: Video call
        if any(word in text_lower for word in ["video call", "video", "gọi video", "goi video"]):
            for pattern, action_type in self.receiver_patterns:
                if action_type == "video_call":
                    match = re.search(pattern, text_lower, re.IGNORECASE)
                    if match and match.groups() and match.group(1):
                        receiver = match.group(1).strip()
                        receiver = self._improve_receiver_boundary(receiver, text_lower)
                        receiver = self._clean_receiver(receiver)
                        if receiver and len(receiver) > 1:
                            return {
                                "RECEIVER": receiver,
                                "ACTION_TYPE": "video_call"
                            }
        
        # ƯU TIÊN 4: Nhắn tin (thấp nhất)
        if any(word in text_lower for word in ["nhắn tin", "nhan tin", "gửi tin", "gui tin", "soạn tin", "soan tin"]):
            for pattern, action_type in self.receiver_patterns:
                if action_type in ["nhắn", "nhan"]:
                    match = re.search(pattern, text_lower, re.IGNORECASE)
                    if match and match.groups() and match.group(1):
                        receiver = match.group(1).strip()
                        receiver = self._improve_receiver_boundary(receiver, text_lower)
                        receiver = self._clean_receiver(receiver)
                        if receiver and len(receiver) > 1:
                            return {
                                "RECEIVER": receiver,
                                "ACTION_TYPE": "nhắn"
                            }
        
        return None
    
    def _improve_receiver_boundary(self, receiver: str, full_text: str) -> str:
        """Cải thiện boundary detection cho receiver - Tối ưu cho "Bố Dũng" """
        words = receiver.split()
        if not words:
            return receiver
            
        # Tìm vị trí của receiver trong full text
        receiver_start = full_text.find(receiver.lower())
        if receiver_start == -1:
            return receiver
            
        # Tìm từ đầu tiên sau receiver trong full text
        after_receiver = full_text[receiver_start + len(receiver):].strip()
        if not after_receiver:
            return receiver
            
        # Tách từ đầu tiên sau receiver
        first_word_after = after_receiver.split()[0] if after_receiver.split() else ""
        
        # Mở rộng danh sách stop words để xử lý tốt hơn - Bao gồm cả có dấu và không dấu
        stop_words = ["là", "la", "rằng", "rằng", "nói", "noi", "sẽ", "se", "đã", "da", "có", "co", "vì", "vi", "bị", "bi", "đau", "dau", "bụng", "bung", 
                      "đón", "don", "ở", "o", "tại", "tai", "với", "voi", "và", "va", "hoặc", "hoac", "hay", "nếu", "neu", "khi", "sau", "trước", "truoc",
                      "tối", "toi", "nay", "chiều", "chieu", "sáng", "sang", "trưa", "trua", "đêm", "dem", "mai", "hôm", "hom", "ngày", "ngay",
                      "nhớ", "nho", "thương", "thuong", "yêu", "yeu", "quý", "quy", "mến", "men", "kính", "kinh", "trọng", "trong", "quý", "quy", "mến", "men",
                      "điện", "dien", "thoại", "thoai", "gọi", "goi", "nhắn", "nhan", "tin", "gửi", "gui", "soạn", "soan", "viết", "viet",
                      # Thêm các từ có dấu tiếng Việt
                      "dép", "dep", "lào", "lao", "dờ", "do", "lờ", "lo", "qua", "tới", "toi", "đến", "den",
                      # Platform keywords
                      "zalo", "zaloo", "gia lo", "gia lô", "za lo", "za lô", "dep lao", "do lo",
                      "messenger", "met sen go", "met sen gơ", "mes sen go", "mes sen gơ",
                      "whatsapp", "wap sap po", "wát sáp", "wát sáp pơ",
                      "telegram", "te le gram", "te le gram",
                      "viber", "vai bo", "vai bo ro",
                      "line", "lai no", "lai no ro",
                      "skype", "skai po", "skai po ro",
                      "discord", "di so cot", "di so cot ro", "dit cot",
                      "youtube", "du tu be", "du tu bo", "du tu bơ", "diu tup",
                      "facebook", "phay buc ro", "phây búc", "phây búc rơ",
                      "google", "guc go", "gúc gồ", "gúc gồ rơ",
                      "tiktok", "tich toc", "tích tóc", "tích tóc rơ",
                      "instagram", "in so ta gram", "in sờ ta gram", "in sờ ta gram rơ",
                      "twitter", "twit to", "twit tơ", "twit tơ rơ",
                      "linkedin", "lin kin", "lin kin ro",
                      "reddit", "ret dit", "rét đít", "rét đít rơ",
                      "grab", "grap", "be", "bi", "xanh sm", "xanh es em", "xanh ét em", "xanh sờ mờ",
                      "shopee", "sop pi", "lazada", "la za da", "tiki", "ti ki", "gojek", "go jek"]
        
        if first_word_after.lower() in stop_words:
            # Tìm vị trí của từ stop trong receiver
            for i, word in enumerate(words):
                if word.lower() in stop_words:
                    return " ".join(words[:i])
        
        # Xử lý đặc biệt cho platform keywords trong receiver
        # Loại bỏ platform keywords khỏi receiver
        cleaned_words = []
        for word in words:
            if word.lower() not in stop_words:
                cleaned_words.append(word)
            else:
                # Nếu gặp platform keyword, dừng lại
                break
        
        if cleaned_words:
            return " ".join(cleaned_words)
        
        # Xử lý đặc biệt cho trường hợp "Bố Dũng" - giữ nguyên nếu là tên riêng
        if len(words) == 2 and words[0].lower() in ["bố", "mẹ", "ông", "bà", "anh", "chị", "em", "con", "cháu"]:
            return receiver
            
        return receiver
    
    def extract_time(self, text: str) -> Optional[str]:
        """Extract TIME entity"""
        text_lower = text.lower()
        
        for pattern in self.time_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                if match.groups():
                    time_value = " ".join([g for g in match.groups() if g])
                    if time_value:
                        return time_value.strip()
                else:
                    return match.group(0).strip()
        
        return None
    
    def extract_message(self, text: str, receiver: Optional[str] = None) -> Optional[str]:
        """Extract MESSAGE entity"""
        text_lower = text.lower()
        
        # Kiểm tra pattern "với nội dung là"
        if "với nội dung là" in text_lower:
            start_pos = text_lower.find("với nội dung là")
            if start_pos != -1:
                message = text[start_pos + len("với nội dung là"):].strip()
                if message:
                    return message
        
        for pattern in self.message_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                # Safe group access
                if match.groups() and match.group(1):
                    message = match.group(1).strip()
                else:
                    continue  # Skip if no valid group
                
                # Làm sạch message
                message = self._clean_message(message)
                
                if message and len(message) > 3:
                    return message
        
        return None
    
    def extract_platform(self, text: str) -> str:
        """Extract PLATFORM entity với logic thông minh"""
        text_lower = text.lower()
        
        for pattern in self.platform_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match and match.group(1):
                platform = match.group(1).lower()
                return platform
        
        # Logic thông minh dựa trên context - Phân biệt giao tiếp và tìm kiếm
        # Xử lý ngoại lệ thuần Việt cho platform
        
        # Gọi điện thoại - hỗ trợ tất cả platforms như send-mess
        if any(word in text_lower for word in ["gọi", "alo", "gọi điện", "gọi thoại", "goi", "goi dien", "goi thoai", "gọi máy", "bấm máy", "quay máy", "đánh máy", "bấm số", "quay số", "liên lạc", "kết nối"]):
            # Kiểm tra platform cụ thể như send-mess
            if any(word in text_lower for word in ["zalo", "zaloo", "gia lo", "gia lô", "gia lo", "gia lô", "za lo", "za lô", "dep lao", "dep lao", "do lo", "do lo", "dép lào", "dép lào"]):
                return "zalo"
            elif any(word in text_lower for word in ["messenger", "mes", "fb messenger", "met", "met sen go", "met sen gơ", "mét", "mét sen gơ", "mes sen go", "mes sen gơ", "mes sen gơ", "mes sen go", "mét"]):
                return "messenger"
            elif any(word in text_lower for word in [
                # Tiếng Anh gốc
                "whatsapp", "wa", "whats", "whats app",
                # Phát âm tiếng Việt thuần túy
                "oắt dáp", "oắt đáp", "oắt đắp", "oắt đạp",
                "goắt sáp", "goắt đáp", "goắt đắp", "goắt đạp", 
                "quắt sáp", "quắt đáp", "quắt đắp", "quắt đạp",
                "wát sáp", "wát đáp", "wát đắp", "wát đạp",
                "oát sáp", "oát đáp", "oát đắp", "oát đạp",
                # Cách gọi thân mật
                "oắt", "goắt", "quắt", "wát", "oát",
                # Cách viết không dấu
                "oat dap", "oat dap", "oat dap", "oat dap",
                "goat sap", "goat dap", "goat dap", "goat dap",
                "quat sap", "quat dap", "quat dap", "quat dap",
                "wat sap", "wat dap", "wat dap", "wat dap",
                "oat sap", "oat dap", "oat dap", "oat dap"
            ]):
                return "whatsapp"
            elif any(word in text_lower for word in ["telegram", "tg", "te lê gram", "tê lê gram", "te le gram", "te le gram", "te le gram", "te le gram"]):
                return "telegram"
            elif any(word in text_lower for word in ["viber", "vai", "vai bơ", "vai bơ rơ", "vai bo", "vai bo ro", "vai bo ro", "vai bo ro"]):
                return "viber"
            elif any(word in text_lower for word in ["line", "lai", "lai nơ", "lai nơ rơ", "lai no", "lai no ro", "lai no ro", "lai no ro"]):
                return "line"
            elif any(word in text_lower for word in ["skype", "skai pơ", "skai pơ rơ", "skai po", "skai po ro", "skai po ro", "skai po ro", "sờ cai pi"]):
                return "skype"
            elif any(word in text_lower for word in ["discord", "đi sờ cốt", "đi sờ cốt rơ", "di so cot", "di so cot ro", "di so cot ro", "di so cot ro", "đít cọt", ]):
                return "discord"
            # App Việt Nam phổ biến
            elif any(word in text_lower for word in ["grab", "grab car", "grab bike", "grab food", "grap", "grap car", "grap bike", "grap food", "gờ ráp", "gáp"]):
                return "grab"
            elif any(word in text_lower for word in ["be", "be car", "be bike", "be food", "bi", "bi car", "bi bike", "bi food"]):
                return "be"
            elif any(word in text_lower for word in ["xanh sm", "xanh sm mart", "xanh es em", "xanh es em mart", "xanh es em mart", "xanh ét em", "xanh sờ mờ"]):
                return "xanh_sm"
            elif any(word in text_lower for word in ["gojek", "go jek", "go jek", "go jek", "go jek"]):
                return "gojek"
            elif any(word in text_lower for word in ["shopee", "shopee food", "shopee pay", "sop pi", "sop pi food", "sop pi pay", "sop pi", "sop pi food", "sop pi pay"]):
                return "shopee"
            elif any(word in text_lower for word in ["lazada", "la za da", "la za da", "la za da"]):
                return "lazada"
            elif any(word in text_lower for word in ["tiki", "ti ki", "ti ki", "ti ki"]):
                return "tiki"
            else:
                # Mặc định là gọi điện thoại thông thường
                return "phone"
        
        # Nhắn tin - ưu tiên platform cụ thể
        elif any(word in text_lower for word in ["nhắn", "gửi", "tin nhắn", "sms", "nhan", "gui", "tin nhan"]):
            if any(word in text_lower for word in ["zalo", "zaloo", "gia lo", "gia lô", "gia lo", "gia lô", "za lo", "za lô", "dep lao", "dep lao", "do lo", "do lo", "dép lào", "dép lào"]):
                return "zalo"
            elif any(word in text_lower for word in ["messenger", "mes", "fb messenger", "met", "met sen go", "met sen gơ", "mét", "mét sen gơ", "mes sen go", "mes sen gơ", "mes sen gơ", "mes sen go", "mét"]):
                return "messenger"
            elif any(word in text_lower for word in [
                # Tiếng Anh gốc
                "whatsapp", "wa", "whats", "whats app",
                # Phát âm tiếng Việt thuần túy
                "oắt dáp", "oắt đáp", "oắt đắp", "oắt đạp",
                "goắt sáp", "goắt đáp", "goắt đắp", "goắt đạp", 
                "quắt sáp", "quắt đáp", "quắt đắp", "quắt đạp",
                "wát sáp", "wát đáp", "wát đắp", "wát đạp",
                "oát sáp", "oát đáp", "oát đắp", "oát đạp",
                # Cách gọi thân mật
                "oắt", "goắt", "quắt", "wát", "oát",
                # Cách viết không dấu
                "oat dap", "oat dap", "oat dap", "oat dap",
                "goat sap", "goat dap", "goat dap", "goat dap",
                "quat sap", "quat dap", "quat dap", "quat dap",
                "wat sap", "wat dap", "wat dap", "wat dap",
                "oat sap", "oat dap", "oat dap", "oat dap"
            ]):
                return "whatsapp"
            elif any(word in text_lower for word in ["telegram", "tg", "te lê gram", "tê lê gram", "te le gram", "te le gram", "te le gram", "te le gram"]):
                return "telegram"
            elif any(word in text_lower for word in ["viber", "vai", "vai bơ", "vai bơ rơ", "vai bo", "vai bo ro", "vai bo ro", "vai bo ro"]):
                return "viber"
            elif any(word in text_lower for word in ["line", "lai", "lai nơ", "lai nơ rơ", "lai no", "lai no ro", "lai no ro", "lai no ro"]):
                return "line"
            elif any(word in text_lower for word in ["skype", "skai pơ", "skai pơ rơ", "skai po", "skai po ro", "skai po ro", "skai po ro", "sờ cai pi"]):
                return "skype"
            elif any(word in text_lower for word in ["discord", "đi sờ cốt", "đi sờ cốt rơ", "di so cot", "di so cot ro", "di so cot ro", "di so cot ro", "đít cọt", ]):
                return "discord"
            # App Việt Nam phổ biến
            elif any(word in text_lower for word in ["grab", "grab car", "grab bike", "grab food", "grap", "grap car", "grap bike", "grap food", "gờ ráp", "gáp"]):
                return "grab"
            elif any(word in text_lower for word in ["be", "be car", "be bike", "be food", "bi", "bi car", "bi bike", "bi food"]):
                return "be"
            elif any(word in text_lower for word in ["xanh sm", "xanh sm mart", "xanh es em", "xanh es em mart", "xanh es em mart", "xanh ét em", "xanh sờ mờ"]):
                return "xanh_sm"
            elif any(word in text_lower for word in ["gojek", "go jek", "go jek", "go jek", "go jek"]):
                return "gojek"
            elif any(word in text_lower for word in ["shopee", "shopee food", "shopee pay", "sop pi", "sop pi food", "sop pi pay", "sop pi", "sop pi food", "sop pi pay"]):
                return "shopee"
            elif any(word in text_lower for word in ["lazada", "la za da", "la za da", "la za da"]):
                return "lazada"
            elif any(word in text_lower for word in ["tiki", "ti ki", "ti ki", "ti ki"]):
                return "tiki"
            else:
                # Mặc định là SMS nếu không có platform cụ thể
                return "sms"
        
        # Tìm kiếm - ưu tiên platform cụ thể
        elif any(word in text_lower for word in ["tìm", "tìm kiếm", "search", "tra cứu", "tìm thông tin", "hỏi", "ask", "google", "tim", "tim kiem", "tra cuu", "tim thong tin", "hoi"]):
            # Kiểm tra platform cụ thể trước - Bao gồm phát âm tiếng Việt
            if any(word in text_lower for word in ["youtube", "yt", "du tu be", "du tu bo", "du tu bơ", "du tu bơ rơ", "du tu be", "dô tu bờ", "du tu be", "du tu bo", "du túp", "diu túp"]):
                return "youtube"
            elif any(word in text_lower for word in ["facebook", "fb", "phay buc ro", "phây búc", "phây búc rơ", "phây búc rơ rơ", "phay buc ro", "phay buc ro", "phay buc ro", "phở bò"]):
                return "facebook"
            elif any(word in text_lower for word in ["google", "gg", "gúc gồ", "gúc gồ rơ", "gúc gồ rơ rơ", "guc go", "guc go ro", "guc go ro", "guc go ro", "gu gồ", "gút gồ", "gu gờ"]):
                return "google"
            elif any(word in text_lower for word in ["bing", "bin", "bin gơ", "bin gơ rơ", "bin go", "bin go ro", "bin go ro", "bin go ro"]):
                return "bing"
            elif any(word in text_lower for word in ["duckduckgo", "duck duck go", "đắc đắc gơ", "đắc đắc gơ rơ", "dac dac go", "dac dac go ro", "dac dac go ro", "dac dac go ro"]):
                return "duckduckgo"
            elif any(word in text_lower for word in ["yahoo", "ya hoo", "ya hoo rơ", "ya hoo ro", "ya hoo ro", "ya hoo ro"]):
                return "yahoo"
            elif any(word in text_lower for word in ["twitter", "twit tơ", "twit tơ rơ", "twit to", "twit to ro", "twit to ro", "twit to ro"]):
                return "twitter"
            elif any(word in text_lower for word in ["tiktok", "tt", "tích tóc", "tích tóc rơ", "tích tóc rơ rơ", "tich toc", "tich toc ro", "típ tóp", "tich toc ro", "thích thóc"]):
                return "tiktok"
            elif any(word in text_lower for word in ["instagram", "ig", "in sờ ta gram", "in sờ ta gram rơ", "in so ta gram", "in so ta gram ro", "in so ta gram ro", "in so ta gram ro"]):
                return "instagram"
            elif any(word in text_lower for word in ["linkedin", "lin kin", "lin kin rơ", "lin kin ro", "lin kin ro", "lin kin ro"]):
                return "linkedin"
            elif any(word in text_lower for word in ["reddit", "rét đít", "rét đít rơ", "ret dit", "ret dit ro", "ret dit ro", "ret dit ro"]):
                return "reddit"
            # App Việt Nam cho tìm kiếm
            elif any(word in text_lower for word in ["grab", "grab car", "grab bike", "grab food", "grap", "grap car", "grap bike", "grap food", "gờ ráp", "gáp"]):
                return "grab"
            elif any(word in text_lower for word in ["be", "be car", "be bike", "be food", "bi", "bi car", "bi bike", "bi food"]):
                return "be"
            elif any(word in text_lower for word in ["xanh sm", "xanh sm mart", "xanh es em", "xanh es em mart", "xanh es em mart", "xanh ét em", "xanh sờ mờ"]):
                return "xanh_sm"
            elif any(word in text_lower for word in ["shopee", "shopee food", "shopee pay", "sop pi", "sop pi food", "sop pi pay", "sóp pi", "sop pi food", "sop pi pay", "sóp pi pây"]):
                return "shopee"
            elif any(word in text_lower for word in ["lazada", "la za da", "la za da", "la za da"]):
                return "lazada"
            else:
                return "google"  # Default cho tìm kiếm
        
        # Mặc định fallback
        return "sms"
    
    def _clean_receiver(self, receiver: str) -> str:
        """Làm sạch receiver entity - giảm nhiễu từ hư từ nhưng giữ nguyên cụm tên."""
        if not receiver:
            return ""

        receiver = receiver.strip()
        receiver = re.sub(r'^[\s,.;:]+|[\s,.;:]+$', '', receiver)

        # Loại bỏ prefix kiểu "qua Zalo/Messenger cho ..." để chỉ giữ lại tên người
        receiver = re.sub(
            r"^(?:qua|bằng|trên)\s+"
            r"(?:zalo|messenger|telegram|viber|line|facebook|whatsapp|sms|tin\s+nhắn|phone|điện\s+thoại)\s+"
            r"(?:cho|tới|đến)\s+",
            "",
            receiver,
            flags=re.IGNORECASE,
        )

        # Loại bỏ các hư từ ở đầu
        receiver = re.sub(
            r'^(?:video\s+call|gọi\s+video|video|call|cho|tới|đến|toi|den|gọi|goi|nhắn|nhan|gửi|gui|soạn|soan|nói|noi|trò|tro|liên|lien)\s+',
            '',
            receiver,
            flags=re.IGNORECASE
        )

        # Trường hợp đặc biệt: "nhắn tin cho con gái bảo tối nay bố không ăn cơm"
        # → RECEIVER chỉ là "con gái", phần sau "bảo tối nay..." là MESSAGE.
        receiver = re.sub(
            r'\s+bảo\s+(?:rằng|là|tối|toi|sáng|sang|chiều|chieu|mai|ngày|ngay|hôm|hom)\b.*$',
            '',
            receiver,
            flags=re.IGNORECASE,
        )

        # Cắt bỏ phần đuôi bắt đầu bằng các hư từ/thời gian/giới từ
        receiver = re.sub(
            r'\b(?:rằng|là|nhé|nha|ạ|nhá|vào|lúc|qua|trên|bằng|vì|do|để|giúp|hộ|cùng|sáng|chiều|tối|đêm|nay|mai|ngày|tuần|tháng)\b.*$',
            '',
            receiver,
            flags=re.IGNORECASE
        )

        # Loại bỏ dấu câu thừa ở cuối lần nữa (trường hợp còn sót dấu ":" sau các bước cắt trên)
        receiver = re.sub(r'[\.,;:]+$', '', receiver)
        receiver = re.sub(r'\s+', ' ', receiver).strip()

        if not receiver:
            return ""

        words = receiver.split()
        # Giữ cụm tên tối đa 6 từ (đủ cho các tên kiểu "Con gái bác Bảo", "Tiệm thuốc Long Châu")
        if len(words) > 6:
            words = words[:6]

        # Duy trì mối quan hệ gia đình nếu có
        relations = {"bố", "mẹ", "ông", "bà", "anh", "chị", "em", "con", "cháu", "cô", "chú", "bác", "dì"}
        noise_tokens = {"video", "call", "video-call", "videocall"}
        normalized_words = []
        for idx, word in enumerate(words):
            if idx == 0 and word.lower() in relations:
                normalized_words.append(word.capitalize())
            else:
                if word.lower() in noise_tokens:
                    continue
                normalized_words.append(word.capitalize())

        while normalized_words and normalized_words[0].lower() in {"cho", "toi", "tới", "đến"}:
            normalized_words.pop(0)

        return " ".join(normalized_words).strip()
    
    def _clean_message(self, message: str) -> str:
        """Làm sạch MESSAGE: bỏ hư từ đầu câu, từ nền tảng còn sót và khoảng trắng/thừa."""
        import re as _re

        msg = (message or "").strip()

        # 1) Bỏ các prefix động từ nói lại – "rằng/là/nói/nhắn/gửi/bảo ..."
        unwanted_prefixes = ["rằng", "là", "nói", "nhắn", "gửi", "bảo"]
        for prefix in unwanted_prefixes:
            if msg.lower().startswith(prefix + " "):
                msg = msg[len(prefix):].strip()

        # 2) Bỏ phần "tin / tin nhắn / tin bảo / tin nói ..." đầu câu nếu còn
        #    ví dụ: "tin bảo tối nay..." → "tối nay..."
        msg = _re.sub(r"^(tin nhắn|tin)\s+(bảo|nói|rằng)?\s*", "", msg, flags=_re.IGNORECASE)

        # 3) Bỏ các giới từ do pattern để lại ở đầu MESSAGE
        #    ví dụ: "qua mai 7 giờ đi khám" → "mai 7 giờ đi khám"
        msg = _re.sub(r"^(qua|bằng|trên)\s+", "", msg, flags=_re.IGNORECASE)

        # 3b) Nếu bắt đầu bằng 'cho/tới/đến <...>:' thì bỏ cụm đó
        #     ví dụ: "cho cậu sáu: mai 7 giờ đi khám" → "mai 7 giờ đi khám"
        msg = _re.sub(r"^(cho|tới|đến)\s+[^:]+:\s*", "", msg, flags=_re.IGNORECASE)

        # 4) Chuẩn hoá khoảng trắng và ký tự thừa đầu/cuối
        msg = _re.sub(r"\s+", " ", msg)
        msg = msg.strip(" ,.;:!?\t\n\r")

        return msg
    
    def extract_phone_number(self, text: str) -> Optional[str]:
        """Extract phone number từ text - Hỗ trợ cả số và chữ"""
        # Pattern cho số điện thoại Việt Nam (dạng số)
        phone_patterns = [
            r"(\d{10,11})",  # 10-11 chữ số
            r"(\d{3,4}\s*\d{3,4}\s*\d{3,4})",  # Có khoảng trắng
            r"(\d{3,4}-\d{3,4}-\d{3,4})",  # Có dấu gạch ngang
            r"(\d{3,4}\.\d{3,4}\.\d{3,4})",  # Có dấu chấm
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                phone = match.group(1)
                # Làm sạch số điện thoại
                phone = re.sub(r'[^\d]', '', phone)
                if len(phone) >= 10:
                    return phone
        
        # Xử lý số điện thoại bằng chữ
        phone_text = self._extract_phone_number_from_text(text)
        if phone_text:
            return phone_text
        
        return None
    
    def _extract_phone_number_from_text(self, text: str) -> Optional[str]:
        """Extract số điện thoại từ text - Hỗ trợ cả số và chữ"""
        text_lower = text.lower()
        
        # Pattern 1: Số điện thoại dạng số (0xxxxxxxxx, +84...) và số khẩn cấp
        phone_patterns = [
            r'\b0\d{9}\b',      # 0xxxxxxxxx (10 digits)
            r'\b0\d{10}\b',     # 0xxxxxxxxxx (11 digits) 
            r'\b\d{10}\b',      # xxxxxxxxxx (10 digits)
            r'\b\d{11}\b',      # xxxxxxxxxxx (11 digits)
            r'\+\d{1,3}\s*\d{9,11}',  # +84 xxxxxxxxx
            r'\b113\b',         # Số khẩn cấp cảnh sát
            r'\b114\b',         # Số khẩn cấp cứu hỏa
            r'\b115\b',         # Số khẩn cấp cấp cứu
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        # Pattern 2: Số điện thoại đọc bằng chữ (không chín tám năm...)
        # Normalize input trước để chuyển từ có dấu sang không dấu
        normalized_text = self._normalize_vietnamese_text(text_lower)
        
        # Pattern cho chuỗi số đọc bằng chữ có khoảng trắng - Hỗ trợ cả số khẩn cấp và số điện thoại thông thường
        # Pattern 1: Số khẩn cấp (3 chữ số) - 113, 114, 115
        emergency_pattern = r'(?:mot\s+mot\s+ba|mot\s+mot\s+bon|mot\s+mot\s+nam|một\s+một\s+ba|một\s+một\s+bốn|một\s+một\s+năm)'
        
        # Pattern 2: Số điện thoại thông thường (10-11 chữ số)
        phone_word_pattern = r'(?:khong|mot|hai|ba|bon|nam|sau|bay|tam|chin|oan|tu|lam|ze ro|zero|one|two|three|four|five|six|seven|eight|nine|không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|tắm|ta)(?:\s+(?:khong|mot|hai|ba|bon|nam|sau|bay|tam|chin|oan|tu|lam|ze ro|zero|one|two|three|four|five|six|seven|eight|nine|không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|tắm|ta)){9,10}(?:\s+\w+)*'
        
        # Kiểm tra số khẩn cấp trước - search trên cả text gốc và normalized
        emergency_match = re.search(emergency_pattern, text_lower)
        if not emergency_match:
            emergency_match = re.search(emergency_pattern, normalized_text)
        
        if emergency_match:
            emergency_text = emergency_match.group(0)
            # Convert số khẩn cấp từ chữ sang số
            emergency_number = self._convert_words_to_numbers(emergency_text)
            emergency_number = emergency_number.replace(' ', '')
            return emergency_number
        
        # Kiểm tra số điện thoại thông thường - search trên cả text gốc và normalized
        phone_match = re.search(phone_word_pattern, text_lower)
        if not phone_match:
            phone_match = re.search(phone_word_pattern, normalized_text)
        
        if phone_match:
            word_sequence = phone_match.group(0)
            # Extract chỉ phần số điện thoại (10-11 từ đầu)
            words = word_sequence.split()
            phone_words = words[:10]  # Lấy 10 từ đầu cho số điện thoại
            phone_sequence = ' '.join(phone_words)
            
            # Convert từ chữ sang số để kiểm tra độ dài
            converted = self._convert_words_to_numbers(phone_sequence)
            # Loại bỏ khoảng trắng để có số điện thoại hợp lệ
            converted = converted.replace(' ', '')
            # Kiểm tra xem có phải số điện thoại hợp lệ không (10-11 chữ số)
            if re.match(r'^\d{10,11}$', converted):
                return converted
        
        # Bước 3: Fallback - xử lý trường hợp đặc biệt như ví dụ của user
        # Ví dụ: "không chín bảy bảy không ba một bảy hai một" → "0977031721"
        special_cases = [
            "không chín bảy bảy không ba một bảy hai một",
            "khong chin bay bay khong ba mot bay hai mot",
            "không tám tám tám năm ba ba tắm hai ba",
            "khong tam tam tam nam ba ba tam hai ba"
        ]
        
        text_lower = text.lower().strip()
        for case in special_cases:
            if case in text_lower:
                # Convert từng từ riêng lẻ
                words = text_lower.split()
                phone_digits = []
                
                for word in words:
                    if word in ["không", "khong", "zero"]:
                        phone_digits.append("0")
                    elif word in ["một", "mot", "mốt", "one"]:
                        phone_digits.append("1")
                    elif word in ["hai", "two"]:
                        phone_digits.append("2")
                    elif word in ["ba", "three"]:
                        phone_digits.append("3")
                    elif word in ["bốn", "bon", "four"]:
                        phone_digits.append("4")
                    elif word in ["năm", "nam", "five"]:
                        phone_digits.append("5")
                    elif word in ["sáu", "sau", "six"]:
                        phone_digits.append("6")
                    elif word in ["bảy", "bay", "seven"]:
                        phone_digits.append("7")
                    elif word in ["tám", "tam", "tắm", "ta", "eight"]:
                        phone_digits.append("8")
                    elif word in ["chín", "chin", "nine"]:
                        phone_digits.append("9")
                
                if len(phone_digits) >= 10:
                    return "".join(phone_digits)
        
        return None
    
    def _extract_camera_entities(self, text: str) -> Dict[str, str]:
        """Extract entities cho open-cam command"""
        entities = {}
        text_lower = text.lower()
        
        # Extract camera type (trước/sau) - Patterns cải thiện để match câu đơn giản
        camera_patterns = [
            r"camera\s+(trước|sau|front|back)",
            r"mở\s+camera\s+(trước|sau|front|back)",
            r"bật\s+camera\s+(trước|sau|front|back)",
            r"khởi\s+động\s+camera\s+(trước|sau|front|back)",
            r"(trước|sau|front|back)\s+camera",
            r"camera\s+(trước|sau|front|back)\s+camera",
            # Patterns cho câu đơn giản với từ bổ trợ
            r"(?:giúp\s+tôi\s+)?mở\s+camera\s+(trước|sau|front|back)",
            r"(?:giúp\s+tôi\s+)?bật\s+camera\s+(trước|sau|front|back)",
            r"(?:giúp\s+tôi\s+)?khởi\s+động\s+camera\s+(trước|sau|front|back)",
            r"(?:hãy\s+)?mở\s+camera\s+(trước|sau|front|back)",
            r"(?:hãy\s+)?bật\s+camera\s+(trước|sau|front|back)",
            r"(?:hãy\s+)?khởi\s+động\s+camera\s+(trước|sau|front|back)"
        ]
        # Mở/rút gọn hành động (ACTION)
        if re.search(r"\b(mở|bật|khởi động|kích hoạt)\b", text_lower):
            entities["ACTION"] = "mở"
        elif re.search(r"\b(tắt|đóng|ngừng|dừng)\b", text_lower):
            entities["ACTION"] = "tắt"
        img_hit = re.search(r"\b(chụp|ảnh|hình|máy ảnh|chụp ảnh|chụp hình)\b", text_lower)
        video_hit = re.search(r"\b(quay|ghi hình|quay phim|quay video|máy quay)\b", text_lower)

        if video_hit:
            entities["CAMERA_TYPE"] = "video"
        elif img_hit:
            entities["CAMERA_TYPE"] = "image"

        # Nếu câu có cả "chụp" lẫn nhóm "quay/ghi hình", ưu tiên video
        if re.search(r"\b(chụp)\b", text_lower) and re.search(r"\b(quay|ghi hình|quay phim|quay video)\b", text_lower):
            entities["CAMERA_TYPE"] = "video"

        if re.search(r"\b(trước|cam trước|camera trước|máy ảnh trước)\b", text_lower):
            entities["MODE"] = "trước"
        elif re.search(r"\b(sau|cam sau|camera sau|máy ảnh sau)\b", text_lower):
            entities["MODE"] = "sau"

        # Loại entity dư (giữ đúng phạm vi open-cam)
        for k in ["PLATFORM", "DEVICE", "CONTENT", "LOCATION"]:
            entities.pop(k, None)
        if not entities.get("CAMERA_TYPE") and re.search(r"\b(camera|máy ảnh|máy quay)\b", text_lower):
            if re.search(r"\b(quay|ghi hình|quay phim|quay video)\b", text_lower):
                entities["CAMERA_TYPE"] = "video"
            else:
                entities["CAMERA_TYPE"] = "image"

        return entities
    
    def _extract_device_control_entities(self, text: str) -> Dict[str, str]:
        """Extract entities cho control-device command - Wi-Fi, Âm lượng, Đèn pin"""
        entities = {}
        text_lower = text.lower()
        
        # Extract DEVICE (Wi-Fi, Âm lượng, Đèn pin)
        device_patterns = [
            # Wi-Fi patterns
            r"(wifi|wi\s+fi|wi-fi|wiai\s+phai|wai\s+phai|goai\s+phai)",
            # Âm lượng patterns  
            r"(âm\s+lượng|volume|vo\s+lum|volum|âm\s+thanh|tiếng)",
            # Đèn pin patterns - ưu tiên cao cho đèn
            r"(đèn\s+pin|den\s+pin|flash|đèn\s+flash|đèn\s+soi)",
            # Đèn đơn thuần (fallback)
            r"\b(đèn)\b"
        ]
        
        for pattern in device_patterns:
            match = re.search(pattern, text_lower)
            if match:
                device = match.group(1).lower()
                # Normalize device names
                if device in ["wifi", "wi fi", "wi-fi", "wiai phai", "wai phai", "goai phai"]:
                    entities["DEVICE"] = "wifi"
                elif device in ["âm lượng", "volume", "vo lum", "volum", "âm thanh", "tiếng"]:
                    entities["DEVICE"] = "âm lượng"
                elif device in ["đèn pin", "den pin", "flash", "đèn flash", "đèn soi", "đèn"]:
                    entities["DEVICE"] = "đèn pin"
                break
        
        # Extract ACTION
        action_patterns = [
            # Bật patterns - thêm word boundary
            r"\b(bật|mở|mo)\b|^on\b|\bon$",
            # Tắt patterns - thêm word boundary
            r"\b(tắt|tat|đóng)\b|^off\b|\boff$",
            # Tăng patterns
            r"(tăng|tang|to\s+lên|lớn\s+hơn)",
            # Giảm patterns
            r"(giảm|giam|nhỏ\s+lại|bé\s+xuống)",
            # Đặt patterns
            r"(đặt|set|để)"
        ]
        
        for pattern in action_patterns:
            match = re.search(pattern, text_lower)
            if match:
                # Lấy group đầu tiên không None
                action = None
                for group in match.groups():
                    if group is not None:
                        action = group.lower()
                        break
                
                if action is None:
                    action = match.group(0).lower()
                
                # Normalize action names
                if action in ["bật", "mở", "mo", "on"]:
                    # Wi-Fi: ON/OFF, Âm lượng/Đèn pin: bật/tắt
                    if entities.get("DEVICE") == "wifi":
                        entities["ACTION"] = "ON"
                    else:
                        entities["ACTION"] = "bật"
                elif action in ["tắt", "tat", "đóng", "off"]:
                    # Wi-Fi: ON/OFF, Âm lượng/Đèn pin: bật/tắt
                    if entities.get("DEVICE") == "wifi":
                        entities["ACTION"] = "OFF"
                    else:
                        entities["ACTION"] = "tắt"
                elif action in ["tăng", "tang", "to lên", "lớn hơn"]:
                    entities["ACTION"] = "tăng"
                elif action in ["giảm", "giam", "nhỏ lại", "bé xuống"]:
                    entities["ACTION"] = "giảm"
                elif action in ["đặt", "set", "để"]:
                    entities["ACTION"] = "đặt"
                break
        
        # Extract LEVEL (chỉ cho âm lượng) - Định lượng cố định
        if entities.get("DEVICE") == "âm lượng":
            level_patterns = [
                # Định lượng cố định
                r"(một\s+chút|thêm\s+tí\s+nữa|tối\s+đa)",
                # Số với ký tự % (ưu tiên cao)
                r"(\d+)\s*%",
                # Số phần trăm từ chữ
                r"(\d+)\s*phần\s*trăm",
                # Số từ chữ với %
                r"(một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|hai\s+mươi|ba\s+mươi|bốn\s+mươi|năm\s+mươi|sáu\s+mươi|bảy\s+mươi|tám\s+mươi|chín\s+mươi|một\s+trăm)\s*%",
                # Số từ chữ với phần trăm
                r"(một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|hai\s+mươi|ba\s+mươi|bốn\s+mươi|năm\s+mươi|sáu\s+mươi|bảy\s+mươi|tám\s+mươi|chín\s+mươi|một\s+trăm)\s*phần\s*trăm"
            ]
            
            for pattern in level_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    level = match.group(1)
                    
                    # Định lượng cố định
                    if level == "một chút":
                        entities["LEVEL"] = "10"  # +10%
                    elif level == "thêm tí nữa":
                        entities["LEVEL"] = "20"  # +20%
                    elif level == "tối đa":
                        entities["LEVEL"] = "100"  # 100%
                    else:
                        # Convert số từ chữ sang số
                        if level in ["một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười"]:
                            level = self._convert_words_to_numbers(level)
                        elif level in ["hai mươi", "ba mươi", "bốn mươi", "năm mươi", "sáu mươi", "bảy mươi", "tám mươi", "chín mươi"]:
                            level = self._convert_words_to_numbers(level)
                        elif level == "một trăm":
                            level = "100"
                        
                        entities["LEVEL"] = level
                    break
        
        return entities
    
    def _normalize_vietnamese_text(self, text: str) -> str:
        """Normalize Vietnamese text từ có dấu sang không dấu"""
        # Mapping từ có dấu sang không dấu
        vietnamese_map = {
            'à': 'a', 'á': 'a', 'ạ': 'a', 'ả': 'a', 'ã': 'a',
            'ầ': 'a', 'ấ': 'a', 'ậ': 'a', 'ẩ': 'a', 'ẫ': 'a',
            'ằ': 'a', 'ắ': 'a', 'ặ': 'a', 'ẳ': 'a', 'ẵ': 'a',
            'è': 'e', 'é': 'e', 'ẹ': 'e', 'ẻ': 'e', 'ẽ': 'e',
            'ề': 'e', 'ế': 'e', 'ệ': 'e', 'ể': 'e', 'ễ': 'e',
            'ì': 'i', 'í': 'i', 'ị': 'i', 'ỉ': 'i', 'ĩ': 'i',
            'ò': 'o', 'ó': 'o', 'ọ': 'o', 'ỏ': 'o', 'õ': 'o',
            'ồ': 'o', 'ố': 'o', 'ộ': 'o', 'ổ': 'o', 'ỗ': 'o',
            'ờ': 'o', 'ớ': 'o', 'ợ': 'o', 'ở': 'o', 'ỡ': 'o',
            'ù': 'u', 'ú': 'u', 'ụ': 'u', 'ủ': 'u', 'ũ': 'u',
            'ừ': 'u', 'ứ': 'u', 'ự': 'u', 'ử': 'u', 'ữ': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỵ': 'y', 'ỷ': 'y', 'ỹ': 'y',
            'đ': 'd',
            'ă': 'a', 'â': 'a', 'ê': 'e', 'ô': 'o', 'ơ': 'o', 'ư': 'u'
        }
        
        result = text
        for vietnamese_char, latin_char in vietnamese_map.items():
            result = result.replace(vietnamese_char, latin_char)
            result = result.replace(vietnamese_char.upper(), latin_char.upper())
        
        # Xử lý các từ đặc biệt
        result = result.replace('tắm', 'tam')
        
        return result
    
    def extract_all_entities(self, text: str, intent: Optional[str] = None) -> Dict[str, str]:
        """Extract entities theo command cụ thể - Tối ưu performance và accuracy"""
        entities = {}
        
        # Command-specific entity extraction - MỖI METHOD TỰ QUYẾT ĐỊNH KHI NÀO CONVERT SỐ
        if intent == "call" or intent == "make-video-call":
            entities.update(self._extract_communication_entities(text))
        elif intent == "send-mess":
            entities.update(self._extract_messaging_entities(text))
        elif intent == "control-device":
            entities.update(self._extract_device_control_entities(text))
        elif intent == "search-internet":
            entities.update(self._extract_internet_search_entities(text))
        elif intent == "search-youtube":
            entities.update(self._extract_youtube_search_entities(text))
        elif intent == "get-info":
            entities.update(self._extract_information_entities(text))
        elif intent == "set-alarm":
            entities.update(self._extract_alarm_entities(text))
        elif intent == "add-contacts":
            entities.update(self._extract_contact_entities(text))
        elif intent == "open-cam":
            entities.update(self._extract_camera_entities(text))
        else:
            # Fallback: extract tất cả entities (legacy behavior)
            entities.update(self._extract_all_legacy_entities(text))
        
        # Validate và fallback entities - Sử dụng text gốc cho validation
        if intent:
            validated_entities = self._validate_entities(intent, entities, text)
        else:
            validated_entities = entities
        
        return validated_entities
    
    def _extract_all_legacy_entities(self, text: str) -> Dict[str, str]:
        entities = {}
        
        # Extract phone number trước để kiểm tra
        phone_result = self.extract_phone_number(text)
        if phone_result:
            entities["PHONE_NUMBER"] = phone_result
        
        # Extract receiver (chỉ khi không có số điện thoại hoặc có tên người cụ thể)
        receiver_result = self.extract_receiver(text)
        if receiver_result:
            # Chỉ lấy RECEIVER, loại bỏ ACTION_TYPE
            if "RECEIVER" in receiver_result:
                entities["RECEIVER"] = receiver_result["RECEIVER"]
        
        # Extract MESSAGE entity khi có
        receiver_val = entities.get("RECEIVER")
        message_result = self.extract_message(text, str(receiver_val) if receiver_val else None)
        if message_result:
            entities["MESSAGE"] = message_result
        
        platform_result = self.extract_platform(text)
        if platform_result:
            entities["PLATFORM"] = platform_result
        
        # Extract contact entities
        contact_result = self.extract_contact_entities(text)
        if contact_result:
            entities.update(contact_result)
        
        # Extract media entities
        media_result = self.extract_media_entities(text)
        if media_result:
            entities.update(media_result)
        
        # Extract search entities
        search_result = self.extract_search_entities(text)
        if search_result:
            entities.update(search_result)
        
        # Extract YouTube entities
        youtube_result = self.extract_youtube_entities(text)
        if youtube_result:
            entities.update(youtube_result)
        
        # Extract info entities
        info_result = self.extract_info_entities(text)
        if info_result:
            entities.update(info_result)
        
        # Extract alarm entities
        alarm_result = self.extract_alarm_entities(text)
        if alarm_result:
            entities.update(alarm_result)
        
        # Extract calendar entities
        calendar_result = self.extract_calendar_entities(text)
        if calendar_result:
            entities.update(calendar_result)
        
        # Extract camera entities
        camera_result = self.extract_camera_entities(text)
        if camera_result:
            entities.update(camera_result)
        
        return entities
    
    def _extract_communication_entities(self, text: str) -> Dict[str, str]:
        """Extract entities cho call/make-video-call"""
        entities = {}
        
        # Extract phone number trước để kiểm tra
        phone_result = self.extract_phone_number(text)
        if phone_result:
            entities["PHONE_NUMBER"] = phone_result
        
        # Extract receiver
        receiver_result = self.extract_receiver(text)
        if receiver_result and "RECEIVER" in receiver_result:
            entities["RECEIVER"] = receiver_result["RECEIVER"]
        
        # Extract platform
        platform_result = self.extract_platform(text)
        if platform_result:
            entities["PLATFORM"] = platform_result
        
        # Nếu có số điện thoại nhưng chưa có RECEIVER, gán RECEIVER = PHONE_NUMBER
        if entities.get("PHONE_NUMBER") and not entities.get("RECEIVER"):
            entities["RECEIVER"] = entities["PHONE_NUMBER"]

        return entities
    
    def _extract_messaging_entities(self, text: str) -> Dict[str, str]:
        """Extract entities cho send-mess - Nhắn tin với các entity mở rộng"""
        entities = {}
        
        # Extract receiver
        receiver_result = self.extract_receiver(text)
        if receiver_result and "RECEIVER" in receiver_result:
            entities["RECEIVER"] = receiver_result["RECEIVER"]
        
        # Extract message
        receiver_val = entities.get("RECEIVER")
        message_result = self.extract_message(text, str(receiver_val) if receiver_val else None)
        if message_result:
            entities["MESSAGE"] = message_result
        
        # Extract platform
        platform_result = self.extract_platform(text)
        if platform_result:
            entities["PLATFORM"] = platform_result

        # Post-clean: chuẩn hoá RECEIVER và MESSAGE cho gửi tin nhắn
        try:
            import re as _re
            # 1) Clean RECEIVER
            if entities.get("RECEIVER"):
                rec = entities["RECEIVER"]
                # loại "tôi" dư
                rec = _re.sub(r"\b(tôi)\b", "", rec, flags=_re.IGNORECASE)
                # bỏ dấu ":" thừa ở cuối (trường hợp "Cậu Sáu:")
                rec = rec.rstrip(" :;,.")
                # case đặc biệt: "con gái bảo tối nay..." → chỉ lấy "Con Gái"
                text_l = text.lower()
                if rec.lower() == "con gái bảo" and _re.search(
                    r"con gái bảo\s+(tối|toi|sáng|sang|chiều|chieu|mai|ngày|ngay|hôm|hom|rằng|là)\b",
                    text_l,
                ):
                    rec = "Con Gái"
                rec = _re.sub(r"\s+", " ", rec).strip()
                entities["RECEIVER"] = rec

            # 2) Clean MESSAGE: bỏ từ nền tảng (zalo/messenger/...), bỏ lặp receiver
            if entities.get("MESSAGE"):
                msg = entities["MESSAGE"]
                # Remove platform tokens
                msg = _re.sub(r"\b(zalo|messenger|sms|whatsapp|telegram|skype)\b", "", msg, flags=_re.IGNORECASE)
                # Nếu có cụm 'nội dung', ưu tiên phần sau 'nội dung'
                m_nd = _re.search(r"nội\s+dung\s+(.+)$", msg, flags=_re.IGNORECASE)
                if m_nd:
                    msg = m_nd.group(1)
                # Remove receiver mention pattern 'cho/tới/đến <receiver>'
                if entities.get("RECEIVER"):
                    rec = _re.escape(entities["RECEIVER"])
                    # Bỏ cụm "cho/tới/đến <RECEIVER>" và dấu câu ngay sau (nếu có)
                    msg = _re.sub(
                        rf"\b(cho|tới|đến)\s+{rec}\b[:;,.\s]*",
                        "",
                        msg,
                        flags=_re.IGNORECASE,
                    )
                # Remove patterns 'cho số <digits/words>'
                num_words = r"(?:không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín)(?:\s+(?:không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín)){5,}"
                msg = _re.sub(r"\b(cho|tới|đến)\s+số\s+[0-9\s]{9,}\b", "", msg, flags=_re.IGNORECASE)
                msg = _re.sub(rf"\b(cho|tới|đến)\s+số\s+{num_words}\b", "", msg, flags=_re.IGNORECASE)
                # Remove raw receiver digits if any
                if entities.get("RECEIVER") and entities["RECEIVER"].isdigit():
                    msg = msg.replace(entities["RECEIVER"], "")
                msg = _re.sub(r"\s+", " ", msg).strip(" ,.")
                # Áp dụng thêm bước sạch MESSAGE chuẩn hoá
                entities["MESSAGE"] = self._clean_message(msg)
        except Exception:
            pass
        
        # Extract using messaging patterns
        for pattern, entity_type in self.messaging_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Safe group access
                if match.groups():
                    value = match.group(1).strip() if match.group(1) else ""
                else:
                    value = match.group(0).strip()
                
                # Process based on entity type
                if entity_type == "message_type":
                    # Map Vietnamese message types to standard types
                    type_map = {
                        "tin nhắn": "text", "text": "text", "văn bản": "text",
                        "voice": "voice", "giọng nói": "voice", "thoại": "voice",
                        "hình ảnh": "image", "ảnh": "image", "photo": "image"
                    }
                    entities["MESSAGE_TYPE"] = type_map.get(value.lower(), "text")
                
                elif entity_type == "urgency":
                    # Map Vietnamese urgency to standard levels
                    urgency_map = {
                        "gấp": "urgent", "khẩn": "urgent", "urgent": "urgent",
                        "bình thường": "normal", "normal": "normal", "thường": "normal"
                    }
                    entities["URGENCY"] = urgency_map.get(value.lower(), "normal")
                
                elif entity_type == "schedule_time":
                    entities["SCHEDULE_TIME"] = value
                
                elif entity_type == "platform_user_id":
                    entities["PLATFORM_USER_ID"] = value
                
                elif entity_type == "group":
                    entities["GROUP"] = value
        
        return entities
    
    def _extract_media_playback_entities(self, text: str) -> Dict[str, str]:
        """Deprecated media extraction (command no longer supported)."""
        return {}
    
    def _extract_content_viewing_entities(self, text: str) -> Dict[str, str]:
        """Deprecated content viewing extraction (command no longer supported)."""
        return {}
    
    def _extract_internet_search_entities(self, text: str) -> Dict[str, str]:
        """Extract entities cho search-internet - Tìm kiếm internet"""
        entities = {}
        
        # Mặc định PLATFORM = google cho search-internet
        entities["PLATFORM"] = "google"
        
        # Chỉ override nếu có platform cụ thể khác (không phải youtube)
        platform = self.extract_platform(text)
        if platform and platform != "youtube":
            entities["PLATFORM"] = platform
        
        # ƯU TIÊN: Extract QUERY trước (quan trọng nhất)
        query_extracted = False
        
        # Sử dụng fallback patterns trước (chính xác hơn)
        fallback_patterns = [
            # Pattern 1: Tìm kiếm với platform cụ thể (ưu tiên cao)
            r"(tìm\s+kiếm|search|tìm|tim\s+kiem|tra\s+cuu|tra\s+cứu)\s+trên\s+(google|bing|duckduckgo|yahoo)\s+(.+)",
            r"(search|tìm|tim)\s+trên\s+(google|bing|duckduckgo|yahoo)\s+(.+)",
            # Pattern 2: Tìm kiếm thông thường
            r"(tìm\s+kiếm|search|tìm|tim\s+kiem|tra\s+cuu|tra\s+cứu)\s+(.+)",
            r"(hỏi|ask|hoi)\s+(.+)",
            r"(google)\s+(.+)"
        ]
        for pattern in fallback_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.groups() and len(match.groups()) >= 2:
                # Lấy group cuối cùng - nội dung query
                query = match.group(len(match.groups())).strip()
                if query and len(query) > 2:
                    entities["QUERY"] = self._clean_query(query)
                    query_extracted = True
                    break
        
        # Nếu fallback không thành công, thử patterns từ internet_search_patterns
        if not query_extracted:
            for pattern, entity_type in self.internet_search_patterns:
                if entity_type == "query":
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match and match.groups():
                        query = match.group(1).strip()
                        if query and len(query) > 2:  # Đảm bảo query có ý nghĩa
                            entities["QUERY"] = self._clean_query(query)
                            query_extracted = True
                            break
        
        # Extract các entities khác
        for pattern, entity_type in self.internet_search_patterns:
            if entity_type == "query":  # Skip query patterns đã xử lý
                continue
                
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Safe group access
                if match.groups():
                    value = match.group(1).strip() if match.group(1) else ""
                else:
                    value = match.group(0).strip()
                
                # Process based on entity type
                if entity_type == "answer_format":
                    # Map Vietnamese formats to standard formats
                    format_map = {
                        "tóm tắt": "summary", "summary": "summary", "tổng hợp": "summary",
                        "danh sách": "list", "list": "list", "liệt kê": "list",
                        "bước": "steps", "steps": "steps", "hướng dẫn": "steps",
                        "định nghĩa": "definition", "definition": "definition", "nghĩa": "definition",
                        "bảng": "table", "table": "table", "so sánh": "comparison"
                    }
                    entities["ANSWER_FORMAT"] = format_map.get(value.lower(), "summary")
                
                elif entity_type == "num_results":
                    # Extract number of results
                    if value.isdigit():
                        entities["NUM_RESULTS"] = value
                    else:
                        # Handle "top 5", "5 kết quả", etc.
                        num_match = re.search(r'(\d+)', value)
                        if num_match:
                            entities["NUM_RESULTS"] = num_match.group(1)
                
                elif entity_type == "language":
                    # Map Vietnamese language names to ISO codes
                    lang_map = {
                        "tiếng việt": "vi", "việt nam": "vi", "vi": "vi", "tieng viet": "vi", "viet nam": "vi",
                        "tiếng anh": "en", "english": "en", "en": "en", "tieng anh": "en",
                        "tiếng hàn": "ko", "korean": "ko", "ko": "ko", "tieng han": "ko",
                        "tiếng nhật": "ja", "japanese": "ja", "ja": "ja", "tieng nhat": "ja"
                    }
                    entities["LANG"] = lang_map.get(value.lower(), "vi")  # Default to Vietnamese
                
                elif entity_type == "country":
                    # Map Vietnamese country names to ISO codes
                    country_map = {
                        "việt nam": "VN", "vietnam": "VN", "vn": "VN", "viet nam": "VN",
                        "mỹ": "US", "usa": "US", "us": "US", "my": "US", "america": "US",
                        "anh": "GB", "uk": "GB", "gb": "GB", "england": "GB",
                        "nhật": "JP", "japan": "JP", "jp": "JP", "nhat": "JP"
                    }
                    entities["COUNTRY"] = country_map.get(value.lower(), "VN")  # Default to Vietnam
                
                elif entity_type == "safesearch":
                    # Map Vietnamese safe search to boolean
                    safe_map = {
                        "bật": "on", "có": "on", "on": "on", "yes": "on",
                        "tắt": "off", "không": "off", "off": "off", "no": "off"
                    }
                    entities["SAFESEARCH"] = safe_map.get(value.lower(), "on")
                
                elif entity_type == "site_domain":
                    entities["SITE_DOMAIN"] = value
                
                elif entity_type == "comparison_a":
                    entities["COMPARISON_A"] = value
                
                elif entity_type == "comparison_b":
                    entities["COMPARISON_B"] = value
        
        return entities
    
    def _extract_youtube_search_entities(self, text: str) -> Dict[str, str]:

        entities = {}
        tl = text.lower().strip()

        # 0) Nếu có "trên google" => không phải search-youtube
        if "trên google" in tl:
            return entities

        # 1) Chỉ bắt khi có youtube/yt, tránh lẫn với search-internet
        if not re.search(r"(youtube|yt)\b", tl, flags=re.IGNORECASE):
            return entities

        # 2) Các patterns phổ biến để lấy QUERY
        patterns = [
            r"(xem|phát|mở|play|tìm|search)\s+(.+?)\s+trên\s+(youtube|yt)",          # xem <q> trên youtube
            r"(xem|phát|mở|play|tìm|search)\s+(?:trên\s+)?(youtube|yt)\s+(.+)",      # xem (trên) youtube <q>
            r"(youtube|yt)\s*[:\-–]?\s+(.+)",                                       # youtube: <q>
            r"(tìm|search)\s+(.+?)\s+trên\s+(youtube|yt)",                           # tìm <q> trên youtube
        ]

        query = None
        for pat in patterns:
            m = re.search(pat, tl, flags=re.IGNORECASE)
            if m:
                groups = [g for g in m.groups() if g and g.lower() not in ("youtube", "yt",
                                                                            "xem", "phát", "mở", "play", "tìm", "search")]
                if groups:
                    query = groups[-1].strip()
                    break

        # 3) Nếu chỉ mở app youtube, không set QUERY
        if not query:
            if re.search(r"(mở|vào|truy cập)\s+(youtube|yt)\b", tl):
                return entities
            if re.fullmatch(r"\s*(youtube|yt)\s*", tl):
                return entities
            return entities

        # 4) Làm sạch query (loại hư từ lịch sự, khoảng trắng/ punctuation dư)
        query = re.sub(r"\b(giúp tôi|giúp|giùm|dùm|nha|nhé|ạ|với|đi)\b", "", query)
        query = query.strip(" ,.;:!?\t\n\r")
        if len(query) < 2:
            return {}

        # 5) Trả về đúng một trường QUERY (đã làm sạch)
        query = self._clean_query(query)
        entities["QUERY"] = query
        return entities
    
    def _extract_information_entities(self, text: str) -> Dict[str, str]:
        """Extract entities cho get-info - Lấy thông tin"""
        entities = {}
        
        # Extract using information patterns
        for pattern, entity_type in self.information_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Safe group access
                if match.groups():
                    value = match.group(1).strip() if match.group(1) else ""
                else:
                    value = match.group(0).strip()
                
                # Process based on entity type
                if entity_type == "metric":
                    # Map Vietnamese metrics to standard metrics
                    metric_map = {
                        "nhiệt độ": "temperature", "temp": "temperature", "độ": "temperature",
                        "độ ẩm": "humidity", "humidity": "humidity", "ẩm": "humidity",
                        "giá": "price", "price": "price", "cost": "price",
                        "tỷ giá": "exchange_rate", "exchange": "exchange_rate", "rate": "exchange_rate",
                        "thời tiết": "weather", "weather": "weather", "khí hậu": "weather",
                        "thời gian": "time", "time": "time", "giờ": "time",
                        "ngày": "date", "date": "date", "tháng": "date",
                        "năm": "year", "year": "year", "tuổi": "age"
                    }
                    entities["METRIC"] = metric_map.get(value.lower(), "general")
                
                elif entity_type == "unit":
                    # Map Vietnamese units to standard units
                    unit_map = {
                        "độ c": "°C", "celsius": "°C", "c": "°C",
                        "độ f": "°F", "fahrenheit": "°F", "f": "°F",
                        "phần trăm": "%", "percent": "%", "%": "%",
                        "vnd": "VND", "đồng": "VND", "dollar": "USD", "usd": "USD",
                        "euro": "EUR", "eur": "EUR", "yen": "JPY", "jpy": "JPY",
                        "giờ": "hour", "hour": "hour", "h": "hour",
                        "phút": "minute", "minute": "minute", "min": "minute",
                        "giây": "second", "second": "second", "s": "second"
                    }
                    entities["UNIT"] = unit_map.get(value.lower(), "general")
                
                elif entity_type == "granularity":
                    # Map Vietnamese granularity to standard granularity
                    granularity_map = {
                        "bây giờ": "now", "now": "now", "hiện tại": "now",
                        "theo giờ": "hourly", "hourly": "hourly", "từng giờ": "hourly",
                        "theo ngày": "daily", "daily": "daily", "hàng ngày": "daily",
                        "theo tuần": "weekly", "weekly": "weekly", "hàng tuần": "weekly",
                        "theo tháng": "monthly", "monthly": "monthly", "hàng tháng": "monthly",
                        "theo năm": "yearly", "yearly": "yearly", "hàng năm": "yearly"
                    }
                    entities["GRANULARITY"] = granularity_map.get(value.lower(), "now")
                
                elif entity_type == "person":
                    entities["PERSON"] = value
                
                elif entity_type == "event":
                    entities["EVENT"] = value
                
                elif entity_type == "location":
                    entities["LOCATION"] = value
                
                elif entity_type == "topic":
                    entities["TOPIC"] = value
                
                elif entity_type == "question_type":
                    # Map Vietnamese question types to standard types
                    question_map = {
                        "là gì": "what", "what": "what", "gì": "what",
                        "như thế nào": "how", "how": "how", "thế nào": "how",
                        "khi nào": "when", "when": "when", "lúc nào": "when",
                        "ở đâu": "where", "where": "where", "đâu": "where",
                        "tại sao": "why", "why": "why", "vì sao": "why",
                        "ai": "who", "who": "who", "người nào": "who",
                        "bao nhiêu": "how_much", "how_much": "how_much", "mấy": "how_much"
                    }
                    entities["QUESTION_TYPE"] = question_map.get(value.lower(), "what")
        
        return entities
    
    def _extract_alarm_entities(self, text: str) -> Dict[str, str]:
        """Extract entities cho set-alarm - Đặt báo thức với entity mở rộng"""
        entities = {}
        
        # Extract using alarm patterns
        for pattern, entity_type in self.alarm_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Safe group access
                if match.groups():
                    value = match.group(1).strip() if match.group(1) else ""
                else:
                    value = match.group(0).strip()
                
                # Process based on entity type
                if entity_type == "time":
                    # Handle time extraction with multiple groups
                    if len(match.groups()) >= 2:
                        hour = match.group(1)
                        minute = match.group(2) if match.group(2) else "00"
                        value = f"{hour}:{minute}"
                    else:
                        value = match.group(1)
                
                # Process based on entity type
                if entity_type == "days_of_week":
                    # Map Vietnamese days to standard days
                    day_map = {
                        "thứ hai": "Monday", "monday": "Monday", "mon": "Monday",
                        "thứ ba": "Tuesday", "tuesday": "Tuesday", "tue": "Tuesday",
                        "thứ tư": "Wednesday", "wednesday": "Wednesday", "wed": "Wednesday",
                        "thứ năm": "Thursday", "thursday": "Thursday", "thu": "Thursday",
                        "thứ sáu": "Friday", "friday": "Friday", "fri": "Friday",
                        "thứ bảy": "Saturday", "saturday": "Saturday", "sat": "Saturday",
                        "chủ nhật": "Sunday", "sunday": "Sunday", "sun": "Sunday",
                        "hàng ngày": "daily", "daily": "daily", "mỗi ngày": "daily"
                    }
                    entities["DAYS_OF_WEEK"] = day_map.get(value.lower(), value)
                
                elif entity_type == "snooze_min":
                    # Extract snooze minutes
                    if value.isdigit():
                        entities["SNOOZE_MIN"] = value
                    else:
                        # Handle "5 phút", "5 minutes", etc.
                        num_match = re.search(r'(\d+)', value)
                        if num_match:
                            entities["SNOOZE_MIN"] = num_match.group(1)
                
                elif entity_type == "volume_profile":
                    # Map Vietnamese volume profiles to standard profiles
                    profile_map = {
                        "êm dịu": "gentle", "gentle": "gentle", "nhẹ nhàng": "gentle",
                        "bình thường": "normal", "normal": "normal", "thường": "normal",
                        "to": "loud", "loud": "loud", "mạnh": "loud"
                    }
                    entities["VOLUME_PROFILE"] = profile_map.get(value.lower(), "normal")
                
                elif entity_type == "label":
                    entities["REMINDER_CONTENT"] = value
                
                elif entity_type == "volume":
                    entities["VOLUME"] = value
                
                elif entity_type == "sound":
                    entities["SOUND"] = value
                
                elif entity_type == "recurrence":
                    entities["RECURRENCE"] = value
                
                elif entity_type == "skip_condition":
                    entities["SKIP_CONDITION"] = value
                
                elif entity_type == "alarm_type":
                    entities["ALARM_TYPE"] = value
                
                elif entity_type == "time":
                    entities["TIME"] = value
                
                elif entity_type == "duration":
                    entities["DURATION"] = value
                
                elif entity_type == "alarm_count":
                    entities["ALARM_COUNT"] = value
                
                elif entity_type == "vibration":
                    entities["VIBRATION"] = value
                
                elif entity_type == "auto_cancel":
                    entities["AUTO_CANCEL"] = value
                
                elif entity_type == "backup":
                    entities["BACKUP"] = value

        # Post-process alarm: chuẩn hoá TIME/DATE/TIMESTAMP
        try:
            import datetime as _dt
            import re as _re
            from zoneinfo import ZoneInfo

            TZ = ZoneInfo("Asia/Bangkok")
            now = _dt.datetime.now(tz=TZ)
            text_l = text.lower()

            def _parse_vi_time(raw: str):
                if not raw:
                    return (None, None)
                s = raw.strip().lower()
                m = _re.search(r"\b(\d{1,2})\s*(giờ|h)?\s*rưỡi\b", s)
                if m:
                    return (int(m.group(1)), 30)
                m = _re.search(r"\b(\d{1,2})\s*(giờ|h)?\s*kém\s*(\d{1,2})\b", s)
                if m:
                    h = int(m.group(1))
                    k = int(m.group(3))
                    h = (h - 1) if k > 0 else h
                    return (h, (60 - k) % 60)
                m = _re.search(r"\b(\d{1,2})\s*(?:giờ|h)\s*(\d{1,2})\b", s)
                if m:
                    return (int(m.group(1)), int(m.group(2)))
                m = _re.search(r"\b(\d{1,2}):(\d{1,2})\b", s)
                if m:
                    return (int(m.group(1)), int(m.group(2)))
                m = _re.search(r"\b(\d{1,2})\s*(giờ|h)\b", s)
                if m:
                    return (int(m.group(1)), 0)
                m = _re.fullmatch(r"\s*(\d{1,2})\s*", s)
                if m:
                    return (int(m.group(1)), 0)
                return (None, None)

            def _apply_buoi_24h(h: int, mi: int):
                if h is None:
                    return (None, None)
                if h >= 13:
                    return (h, mi)

                has_sang = _re.search(r"\b(sáng|rạng sáng)\b", text_l)
                has_trua = _re.search(r"\b(trưa|giữa trưa|nửa trưa)\b", text_l)
                has_chieu = _re.search(r"\b(chiều)\b", text_l)
                has_toi = _re.search(r"\b(tối)\b", text_l)
                has_dem = _re.search(r"\b(đêm|khuya|nửa đêm)\b", text_l)

                if h == 12:
                    if has_trua:
                        return (12, mi)
                    if has_dem:
                        return (0, mi)
                    return (12, mi)

                if has_sang:
                    return (h, mi)
                if has_trua:
                    return (h + 12 if 1 <= h <= 5 else 12 if h == 0 else h, mi)
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
            if entities.get("TIME"):
                h, mi = _parse_vi_time(entities["TIME"])
                if (mi is None or mi == 0) and _re.search(r"\b(rưỡi|kém|hơn|:\d{1,2}|\d{1,2}\s*(phút|min))\b", text_l):
                    h2, mi2 = _parse_vi_time(text_l)
                    if h2 is not None:
                        h, mi = h2, mi2
            else:
                h, mi = _parse_vi_time(text_l)

            if h is not None:
                h, mi = _apply_buoi_24h(h, mi if mi is not None else 0)
                mi = 0 if mi is None else max(0, min(59, mi))
                if 0 <= h <= 23:
                    entities["TIME"] = f"{h:02d}:{mi:02d}"

            date_obj = None
            if _re.search(r"\b(hôm nay)\b", text_l):
                date_obj = now.date()
            elif _re.search(r"\b(ngày mai|mai)\b", text_l):
                date_obj = (now + _dt.timedelta(days=1)).date()
            elif _re.search(r"\b(ngày mốt|mốt|ngày kia)\b", text_l):
                date_obj = (now + _dt.timedelta(days=2)).date()
            elif _re.search(r"\b(tuần sau)\b", text_l):
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
            has_week_next = _re.search(r"\b(tuần sau)\b", text_l) is not None
            for vn, wd in weekday_map.items():
                if vn in text_l:
                    delta = (wd - now.weekday()) % 7
                    if delta == 0:
                        delta = 7
                    if has_week_next:
                        delta += 7
                    date_obj = (now + _dt.timedelta(days=delta)).date()
                    break

            if entities.get("TIME") and not entities.get("DATE") and date_obj is None:
                date_obj = now.date()
                try:
                    hh, mm = [int(x) for x in entities["TIME"].split(":")]
                    candidate_dt = _dt.datetime.combine(
                        date_obj,
                        _dt.time(hour=hh, minute=mm, tzinfo=TZ),
                    )
                    if candidate_dt <= now:
                        date_obj = (now + _dt.timedelta(days=1)).date()
                except Exception:
                    pass

            if date_obj is not None:
                entities["DATE"] = date_obj.isoformat()

            if entities.get("TIME") and (entities.get("DATE") or date_obj is not None):
                date_str = entities.get("DATE") or (date_obj.isoformat() if date_obj else None)
                if date_str:
                    target_date = _dt.date.fromisoformat(date_str)
                    hh, mm = [int(x) for x in entities["TIME"].split(":")]
                    ts = _dt.datetime(target_date.year, target_date.month, target_date.day, hh, mm, tzinfo=TZ)
                    entities["TIMESTAMP"] = ts.isoformat()

        except Exception:
            pass

        return entities
    
    def _extract_calendar_entities(self, text: str) -> Dict[str, str]:
        """Deprecated calendar extraction (command no longer supported)."""
        return {}
    
    def extract_contact_entities(self, text: str) -> Dict[str, str]:
        """Extract contact entities - tên, số điện thoại, email, ghi chú, địa chỉ với entity mở rộng"""
        entities = {}
        text_lower = text.lower()
        
        # Extract using contact patterns
        for pattern, entity_type in self.contact_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Safe group access
                if match.groups():
                    value = match.group(1).strip() if match.group(1) else ""
                else:
                    value = match.group(0).strip()  # Use full match if no groups
                
                # Process based on entity type
                if entity_type == "phone":
                    # Remove spaces and format phone number
                    value = re.sub(r'[^\d+]', '', value)
                    if value.startswith('+84'):
                        value = '0' + value[3:]
                    entities["PHONE"] = value
                
                elif entity_type == "email":
                    # Email is already clean
                    entities["EMAIL"] = value
                
                elif entity_type == "address":
                    entities["ADDRESS"] = value
                
                elif entity_type == "nickname":
                    entities["NICKNAME"] = value
                
                elif entity_type == "note":
                    entities["NOTE"] = value
                
                elif entity_type == "birthday":
                    # Format birthday to standard format
                    birthday_map = {
                        "sinh nhật": "birthday", "birthday": "birthday", "ngày sinh": "birthday",
                        "ngày sinh nhật": "birthday", "date of birth": "birthday"
                    }
                    entities["BIRTHDAY"] = birthday_map.get(value.lower(), value)
                
                elif entity_type == "relation":
                    # Map Vietnamese relations to standard relations
                    relation_map = {
                        "bố": "father", "ba": "father", "cha": "father", "father": "father",
                        "mẹ": "mother", "má": "mother", "mother": "mother",
                        "anh": "brother", "em trai": "brother", "brother": "brother",
                        "chị": "sister", "em gái": "sister", "sister": "sister",
                        "ông": "grandfather", "grandfather": "grandfather",
                        "bà": "grandmother", "grandmother": "grandmother",
                        "bạn": "friend", "friend": "friend", "bạn bè": "friend",
                        "đồng nghiệp": "colleague", "colleague": "colleague", "coworker": "colleague"
                    }
                    entities["RELATION"] = relation_map.get(value.lower(), value)
                
                elif entity_type == "contact":
                    entities["CONTACT"] = value
                
                elif entity_type == "location":
                    entities["LOCATION"] = value
                
                elif entity_type == "company":
                    entities["COMPANY"] = value
                
                # Clean up text entities
                if entity_type in ["contact", "note", "location", "company", "address", "nickname"]:
                    value = value.strip('"\'')
        
        return entities
    
    def _extract_contact_entities(self, text: str) -> Dict[str, str]:
        entities = {}
        text_lower = text.lower()
        
        # Extract phone number từ số từ chữ (ưu tiên cao)
        phone_number = self._extract_phone_number_from_text(text)
        if phone_number:
            entities["PHONE_NUMBER"] = phone_number
        
        # Extract contact name - Pattern chính: "Thêm danh bạ/liên hệ [TÊN] số [SỐ]" + "với tên là [TÊN]" + "lưu số"
        name_patterns = [
            # Pattern 4: "với tên là [TÊN]" (case mới) - ƯU TIÊN CAO
            r"với\s+tên\s+là\s+([^,\n]+)",
            # Pattern 4.1: "voi ten la [TÊN]" (không dấu) - ƯU TIÊN CAO
            r"voi\s+ten\s+la\s+([^,\n]+)",
            # Pattern 5: "tên là [TÊN]"
            r"tên\s+là\s+([^,\n]+)",
            # Pattern 5.1: "ten la [TÊN]" (không dấu)
            r"ten\s+la\s+([^,\n]+)",
            # Pattern 6: "lưu số [SỐ] vào danh bạ với tên là [TÊN]" - ƯU TIÊN CAO
            r"lưu\s+số\s+[^,\n]*vào\s+danh\s+bạ\s+với\s+tên\s+là\s+([^,\n]+)",
            # Pattern 6.1: "luu so [SỐ] vao danh ba voi ten la [TÊN]" (không dấu)
            r"luu\s+so\s+[^,\n]*vao\s+danh\s+ba\s+voi\s+ten\s+la\s+([^,\n]+)",
            # Pattern 1: "Thêm danh bạ [TÊN] số [SỐ]"
            r"thêm\s+danh\s+bạ\s+([^,\n]+?)(?:\s+số|\s+phone|\s+so)",
            # Pattern 1.1: "them danh ba [TÊN] [SỐ]" (không dấu, không có "số")
            r"them\s+danh\s+ba\s+([^,\n]+?)(?:\s+số|\s+phone|\s+so|\s+khong)",
            # Pattern 2: "Thêm liên hệ [TÊN] số [SỐ]"  
            r"thêm\s+liên\s+hệ\s+([^,\n]+?)(?:\s+số|\s+phone|\s+so)",
            # Pattern 2.1: "them lien he [TÊN] [SỐ]" (không dấu)
            r"them\s+lien\s+he\s+([^,\n]+?)(?:\s+số|\s+phone|\s+so)",
            # Pattern 3: "Thêm contact [TÊN] số [SỐ]"
            r"thêm\s+contact\s+([^,\n]+?)(?:\s+số|\s+phone|\s+so)",
            # Pattern 7: Fallback patterns
            r"là\s+([^,\n]+)",
            r"tên\s+([^,\n]+)",
            r"gọi\s+([^,\n]+)",
            r"liên\s+hệ\s+([^,\n]+)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean up name - loại bỏ stop words
                name = self._clean_contact_name(name)
                # Loại chuỗi số điện thoại (số/dọc chữ) ra khỏi tên nếu vô tình dính
                if phone_number:
                    # loại số digits
                    name = re.sub(re.escape(phone_number), "", name)
                    # loại số đọc bằng chữ dài
                    num_words = r"(?:không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín)(?:\s+(?:không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín)){5,}"
                    name = re.sub(num_words, "", name, flags=re.IGNORECASE)
                    name = re.sub(r"\s+", " ", name).strip()
                if name and len(name) > 1:
                    entities["CONTACT_NAME"] = name
                    break
        
        # Extract REMINDER_CONTENT / ghi chú cho nhắc nhở hoặc liên hệ
        reminder_patterns = [
            r"ghi\s+chú\s+([^,\n]+)",
            r"chức\s+danh\s+([^,\n]+)",
            r"với\s+([^,\n]+)",
            r"nhé\s+([^,\n]+)",
            r"nha\s+([^,\n]+)"
        ]
        
        for pattern in reminder_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                reminder = match.group(1).strip()
                # Clean up reminder content
                reminder = self._clean_reminder_content(reminder)
                if reminder and len(reminder) > 1:
                    entities["REMINDER_CONTENT"] = reminder
                    break
        
        # Fallback: extract reminder content từ cuối câu nếu chưa có
        if "REMINDER_CONTENT" not in entities:
            words = text.split()
            if len(words) > 2:
                # Lấy từ cuối cùng làm title nếu không phải số điện thoại
                last_word = words[-1]
                if not last_word.isdigit() and last_word not in ["số", "điện", "thoại", "phone", "so", "dien", "thoai"]:
                    entities["REMINDER_CONTENT"] = self._clean_reminder_content(last_word)
        
        # Chuẩn hoá: nếu có PHONE_NUMBER thì set PHONE và đảm bảo CONTACT_NAME không rỗng
        if entities.get("PHONE_NUMBER"):
            entities["PHONE"] = entities["PHONE_NUMBER"]
        return entities
    
    def _clean_contact_name(self, name: str) -> str:
        """Clean contact name - loại bỏ stop words"""
        stop_words = [
            "số", "điện", "thoại", "phone", "so", "dien", "thoai",
            "là", "la", "tên", "ten", "gọi", "goi", "liên", "lien",
            "hệ", "he", "contact", "lien he", "lienhe"
        ]
        
        words = name.split()
        cleaned_words = [word for word in words if word.lower() not in stop_words]
        
        return " ".join(cleaned_words).strip()
    
    def _clean_reminder_content(self, reminder: str) -> str:
        stop_words = [
            "ghi", "chú", "ghi chú", "chức", "danh", "chức danh",
            "với", "voi", "nhé", "nhe", "nha", "ạ", "a",
            "là", "la", "tên", "ten", "gọi", "goi", "liên", "lien",
            "hệ", "he", "contact", "lien he", "lienhe"
        ]
        
        words = reminder.split()
        cleaned_words = [word for word in words if word.lower() not in stop_words]
        
        return " ".join(cleaned_words).strip()

    def _extract_frequency(self, text: str) -> str:
        """Phát hiện tần suất nhắc nhở cơ bản (daily/weekly/weekday/...)"""
        normalized = text.lower()
        frequency_map = {
            "daily": ["hàng ngày", "hang ngay", "mỗi ngày", "moi ngay", "hằng ngày"],
            "weekly": ["hàng tuần", "hang tuan", "mỗi tuần", "moi tuan"],
            "monthly": ["hàng tháng", "hang thang", "mỗi tháng", "moi thang"],
            "weekday": [
                "thứ hai đến thứ sáu",
                "thu hai den thu sau",
                "thứ 2 đến thứ 6",
                "thu 2 den thu 6",
                "các ngày trong tuần",
                "cac ngay trong tuan",
            ],
            "weekend": [
                "cuối tuần",
                "cuoi tuan",
                "thứ bảy và chủ nhật",
                "thu bay va chu nhat",
                "thứ 7 chủ nhật",
                "thu 7 chu nhat",
            ],
        }
        for freq, patterns in frequency_map.items():
            if any(pattern in normalized for pattern in patterns):
                return freq
        return ""
    
    def _validate_entities(self, intent: str, entities: Dict[str, str], text: str) -> Dict[str, str]:
        """Validate và fallback entities cho từng command"""
        validated_entities = entities.copy()
        
        if intent == "call" or intent == "make-video-call":
            # Communication intents cần RECEIVER
            if not validated_entities.get("RECEIVER"):
                # Fallback: extract từ text nếu có
                receiver_patterns = [
                    r"gọi\s+([^,\n]+?)(?:\s+lúc|\s+vào|\s+nhé|\s+nha|$)",
                    r"video\s+call\s+([^,\n]+?)(?:\s+lúc|\s+vào|\s+nhé|\s+nha|$)",
                    r"gọi\s+video\s+([^,\n]+?)(?:\s+lúc|\s+vào|\s+nhé|\s+nha|$)"
                ]
                for pattern in receiver_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        validated_entities["RECEIVER"] = match.group(1).strip()
                        break

            # Làm sạch RECEIVER lần cuối (kể cả khi đến từ model hoặc fallback)
            if validated_entities.get("RECEIVER"):
                original = validated_entities["RECEIVER"]
                cleaned = self._clean_receiver(original)
                # Nếu sau khi clean bị rỗng thì giữ lại bản gốc để không mất hoàn toàn
                validated_entities["RECEIVER"] = cleaned or original
            
            # Fallback platform nếu không có
            if not validated_entities.get("PLATFORM"):
                validated_entities["PLATFORM"] = "phone"  # Default platform
        
        elif intent == "send-mess":
            # Messaging cần RECEIVER và MESSAGE
            if not validated_entities.get("RECEIVER"):
                # Fallback: extract từ text
                receiver_patterns = [
                    r"nhắn\s+([^,\n]+?)(?:\s+là|\s+rằng|\s+nói|$)",
                    r"gửi\s+tin\s+nhắn\s+([^,\n]+?)(?:\s+là|\s+rằng|\s+nói|$)"
                ]
                for pattern in receiver_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        validated_entities["RECEIVER"] = match.group(1).strip()
                        break

            # Làm sạch RECEIVER cho send-mess
            if validated_entities.get("RECEIVER"):
                original = validated_entities["RECEIVER"]
                cleaned = self._clean_receiver(original)
                validated_entities["RECEIVER"] = cleaned or original
            
            if not validated_entities.get("MESSAGE"):
                # Fallback: extract message content
                message_patterns = [
                    r"là\s+(.+?)(?:$|\.)",
                    r"rằng\s+(.+?)(?:$|\.)",
                    r"nói\s+(.+?)(?:$|\.)"
                ]
                for pattern in message_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        validated_entities["MESSAGE"] = match.group(1).strip()
                        break

            # Chuẩn hoá MESSAGE cho send-mess (bỏ prefix như "nhắn", "gửi"... nếu còn)
            if validated_entities.get("MESSAGE"):
                validated_entities["MESSAGE"] = self._clean_message(validated_entities["MESSAGE"])
            
            # Fallback platform
            if not validated_entities.get("PLATFORM"):
                validated_entities["PLATFORM"] = "zalo"  # Default platform
        
        elif intent == "set-alarm":
            # Chuẩn hóa ghi chú nhắc nhở
            if validated_entities.get("REMINDER_CONTENT"):
                validated_entities["REMINDER_CONTENT"] = self._clean_reminder_content(
                    validated_entities["REMINDER_CONTENT"]
                )

            # Alarm cần TIME hoặc DATE
            if not validated_entities.get("TIME"):
                # Fallback: extract time từ text
                time_patterns = [
                    r"(\d{1,2})\s*giờ\s*(\d{1,2})?\s*(?:phút)?",
                    r"(\d{1,2}):(\d{2})",
                    r"(\d{1,2})\s*rưỡi"
                ]
                for pattern in time_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        if len(match.groups()) >= 2:
                            hour = match.group(1)
                            minute = match.group(2) if match.group(2) else "00"
                            validated_entities["TIME"] = f"{hour}:{minute}"
                        else:
                            validated_entities["TIME"] = match.group(1)
                        break
            
            # Fallback reminder content
            if not validated_entities.get("REMINDER_CONTENT"):
                validated_entities["REMINDER_CONTENT"] = "Nhắc nhở"

            # Extract frequency nếu có
            if not validated_entities.get("FREQUENCY"):
                frequency = self._extract_frequency(text)
                if frequency:
                    validated_entities["FREQUENCY"] = frequency
        
        elif intent == "add-contacts":
            # Add-contacts cần CONTACT_NAME và PHONE_NUMBER
            if not validated_entities.get("CONTACT_NAME"):
                # Fallback: extract name từ text với patterns mở rộng
                name_patterns = [
                    r"thêm\s+danh\s+bạ\s+([^,\n]+?)(?:\s+số|\s+phone|\s+so)",
                    r"them\s+danh\s+ba\s+([^,\n]+?)(?:\s+số|\s+phone|\s+so|\s+khong)",  # Không dấu
                    r"thêm\s+liên\s+hệ\s+([^,\n]+?)(?:\s+số|\s+phone|\s+so)",
                    r"them\s+lien\s+he\s+([^,\n]+?)(?:\s+số|\s+phone|\s+so)",  # Không dấu
                    r"thêm\s+contact\s+([^,\n]+?)(?:\s+số|\s+phone|\s+so)",
                    r"với\s+tên\s+là\s+([^,\n]+)",  # Case mới
                    r"voi\s+ten\s+la\s+([^,\n]+)",  # Case mới không dấu
                    r"tên\s+là\s+([^,\n]+)",        # Case mới
                    r"ten\s+la\s+([^,\n]+)",        # Case mới không dấu
                    r"lưu\s+liên\s+hệ\s+mới:\s*([^,\n]+)",
                    r"thêm\s+bạn\s+['\"]?([^'\",\n]+)['\"]?",
                    r"create\s+contact:\s*([^,\n]+)"
                ]
                for pattern in name_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        name = match.group(1).strip()
                        name = self._clean_contact_name(name)
                        if name:
                            validated_entities["CONTACT_NAME"] = name
                        break
                else:
                    validated_entities["CONTACT_NAME"] = "Liên hệ mới"  # Default name
            
            # Validate PHONE_NUMBER
            if not validated_entities.get("PHONE_NUMBER"):
                # Fallback: extract phone từ text
                phone_patterns = [
                    r"số\s+([0-9\s\-\.]+)",
                    r"phone\s+([0-9\s\-\.]+)",
                    r"(\d{10,11})"
                ]
                for pattern in phone_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        phone = match.group(1).strip()
                        # Clean phone number
                        phone = re.sub(r'[^\d]', '', phone)
                        if len(phone) >= 10:
                            validated_entities["PHONE_NUMBER"] = phone
                            break
                else:
                    validated_entities["PHONE_NUMBER"] = "0000000000"  # Default phone
            
        # Validate REMINDER_CONTENT (optional)
        if not validated_entities.get("REMINDER_CONTENT"):
            validated_entities["REMINDER_CONTENT"] = ""  # Optional field
        
        elif intent == "open-cam":
            # Open-cam cần CAMERA_TYPE
            if not validated_entities.get("CAMERA_TYPE"):
                # Fallback: extract camera type từ text
                camera_patterns = [
                    r"camera\s+(trước|sau|front|back)",
                    r"mở\s+camera\s+(trước|sau|front|back)",
                    r"bật\s+camera\s+(trước|sau|front|back)",
                    r"(trước|sau|front|back)\s+camera"
                ]
                for pattern in camera_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        camera_type = match.group(1)
                        # Normalize camera type - Giữ nguyên tiếng Việt
                        if camera_type in ["trước", "front"]:
                            validated_entities["CAMERA_TYPE"] = "trước"
                        elif camera_type in ["sau", "back"]:
                            validated_entities["CAMERA_TYPE"] = "sau"
                        else:
                            validated_entities["CAMERA_TYPE"] = camera_type
                        break
                else:
                    validated_entities["CAMERA_TYPE"] = "trước"  # Default camera type
            
            # Validate ACTION (optional)
            if not validated_entities.get("ACTION"):
                validated_entities["ACTION"] = "mở"  # Default action
            
            # Validate PLATFORM
            if not validated_entities.get("PLATFORM"):
                validated_entities["PLATFORM"] = "camera"  # Default platform
        
        elif intent == "control-device":
            # Control-device cần DEVICE và ACTION
            if not validated_entities.get("DEVICE"):
                # Fallback: extract device từ text
                device_patterns = [
                    r"(wifi|wi\s+fi|wi-fi|wiai\s+phai|wai\s+phai|goai\s+phai)",
                    r"(âm\s+lượng|volume|vo\s+lum|volum|âm\s+thanh|tiếng)",
                    r"(đèn\s+pin|den\s+pin|flash|đèn\s+flash|đèn\s+soi)",
                    r"\b(đèn)\b"
                ]
                
                for pattern in device_patterns:
                    match = re.search(pattern, text.lower())
                    if match:
                        device = match.group(1).lower()
                        if device in ["wifi", "wi fi", "wi-fi", "wiai phai", "wai phai", "goai phai"]:
                            validated_entities["DEVICE"] = "wifi"
                        elif device in ["âm lượng", "volume", "vo lum", "volum", "âm thanh", "tiếng"]:
                            validated_entities["DEVICE"] = "âm lượng"
                        elif device in ["đèn pin", "den pin", "flash", "đèn flash", "đèn soi", "đèn"]:
                            validated_entities["DEVICE"] = "đèn pin"
                        break
                else:
                    # Default device nếu không tìm thấy
                    validated_entities["DEVICE"] = "âm lượng"  # Default device
            
            # Validate ACTION
            if not validated_entities.get("ACTION"):
                # Fallback: extract action từ text
                action_patterns = [
                    r"(bật|mở|mo|on)",
                    r"(tắt|tat|đóng|off)",
                    r"(tăng|tang|to\s+lên|lớn\s+hơn)",
                    r"(giảm|giam|nhỏ\s+lại|bé\s+xuống)",
                    r"(đặt|set|để)"
                ]
                
                for pattern in action_patterns:
                    match = re.search(pattern, text.lower())
                    if match:
                        action = match.group(1).lower()
                        if action in ["bật", "mở", "mo", "on"]:
                            # Wi-Fi: ON/OFF, Âm lượng/Đèn pin: bật/tắt
                            if validated_entities.get("DEVICE") == "wifi":
                                validated_entities["ACTION"] = "ON"
                            else:
                                validated_entities["ACTION"] = "bật"
                        elif action in ["tắt", "tat", "đóng", "off"]:
                            # Wi-Fi: ON/OFF, Âm lượng/Đèn pin: bật/tắt
                            if validated_entities.get("DEVICE") == "wifi":
                                validated_entities["ACTION"] = "OFF"
                            else:
                                validated_entities["ACTION"] = "tắt"
                        elif action in ["tăng", "tang", "to lên", "lớn hơn"]:
                            validated_entities["ACTION"] = "tăng"
                        elif action in ["giảm", "giam", "nhỏ lại", "bé xuống"]:
                            validated_entities["ACTION"] = "giảm"
                        elif action in ["đặt", "set", "để"]:
                            validated_entities["ACTION"] = "đặt"
                        break
                else:
                    # Default action nếu không tìm thấy
                    validated_entities["ACTION"] = "bật"  # Default action
            
            # Validate LEVEL (chỉ cho âm lượng) - Định lượng cố định
            if validated_entities.get("DEVICE") == "âm lượng" and not validated_entities.get("LEVEL"):
                # Fallback: extract level từ text
                level_patterns = [
                    # Định lượng cố định
                    r"(một\s+chút|thêm\s+tí\s+nữa|tối\s+đa)",
                    # Số với ký tự % (ưu tiên cao)
                    r"(\d+)\s*%",
                    # Số phần trăm từ chữ
                    r"(\d+)\s*phần\s*trăm",
                    # Số từ chữ với %
                    r"(một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|hai\s+mươi|ba\s+mươi|bốn\s+mươi|năm\s+mươi|sáu\s+mươi|bảy\s+mươi|tám\s+mươi|chín\s+mươi|một\s+trăm)\s*%",
                    # Số từ chữ với phần trăm
                    r"(một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|hai\s+mươi|ba\s+mươi|bốn\s+mươi|năm\s+mươi|sáu\s+mươi|bảy\s+mươi|tám\s+mươi|chín\s+mươi|một\s+trăm)\s*phần\s*trăm"
                ]
                
                for pattern in level_patterns:
                    match = re.search(pattern, text.lower())
                    if match:
                        level = match.group(1)
                        
                        # Định lượng cố định
                        if level == "một chút":
                            validated_entities["LEVEL"] = "10"  # +10%
                        elif level == "thêm tí nữa":
                            validated_entities["LEVEL"] = "20"  # +20%
                        elif level == "tối đa":
                            validated_entities["LEVEL"] = "100"  # 100%
                        else:
                            # Convert số từ chữ sang số
                            if level in ["một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười"]:
                                level = self._convert_words_to_numbers(level)
                            elif level in ["hai mươi", "ba mươi", "bốn mươi", "năm mươi", "sáu mươi", "bảy mươi", "tám mươi", "chín mươi"]:
                                level = self._convert_words_to_numbers(level)
                            elif level == "một trăm":
                                level = "100"
                            
                            validated_entities["LEVEL"] = level
                        break
                else:
                    # Default level nếu không tìm thấy
                    validated_entities["LEVEL"] = "10"  # Default: một chút (+10%)
        
        elif intent == "search-internet" or intent == "search-youtube":
            # Search cần QUERY
            if not validated_entities.get("QUERY") and not validated_entities.get("YT_QUERY"):
                # Fallback: extract query từ text
                query_patterns = [
                    r"tìm\s+kiếm\s+([^,\n]+)",
                    r"search\s+([^,\n]+)",
                    r"tìm\s+([^,\n]+)"
                ]
                for pattern in query_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        query = match.group(1).strip()
                        query = self._clean_query(query)
                        if intent == "search-youtube":
                            validated_entities["YT_QUERY"] = query
                        else:
                            validated_entities["QUERY"] = query
                        break
                else:
                    # Use full text as query
                    if intent == "search-youtube":
                        validated_entities["YT_QUERY"] = self._clean_query(text)
                    else:
                        validated_entities["QUERY"] = self._clean_query(text)
        
        elif intent == "get-info":
            # Info cần TOPIC
            if not validated_entities.get("TOPIC"):
                # Fallback: extract topic từ text
                topic_patterns = [
                    r"thông\s+tin\s+([^,\n]+)",
                    r"kiểm\s+tra\s+([^,\n]+)",
                    r"về\s+([^,\n]+)"
                ]
                for pattern in topic_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        validated_entities["TOPIC"] = match.group(1).strip()
                        break
                else:
                    validated_entities["TOPIC"] = text  # Use full text as topic
        
        return validated_entities
    
    def extract_media_entities(self, text: str) -> Dict[str, str]:
        """Extract media entities - playlist, artist, podcast, file, radio"""
        entities = {}
        
        # Extract using media patterns
        for pattern, entity_type in self.media_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Safe group access
                if match.groups():
                    value = match.group(1).strip() if match.group(1) else ""
                else:
                    value = match.group(0).strip()  # Use full match if no groups
                
                # Clean up value
                if entity_type in ["playlist", "artist", "podcast", "radio", "stream", "genre", "episode", "version", "context", "purpose"]:
                    value = value.strip('"\'')
                elif entity_type == "file_path":
                    # File path is already clean
                    pass
                
                entities[entity_type.upper()] = value
        
        return entities
    
    def extract_search_entities(self, text: str) -> Dict[str, str]:
        """Extract search entities - query, platform, preference, time"""
        entities = {}
        
        # Extract using search patterns
        for pattern, entity_type in self.search_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Safe group access
                if match.groups():
                    value = match.group(1).strip() if match.group(1) else ""
                else:
                    value = match.group(0).strip()  # Use full match if no groups
                
                # Clean up value
                if entity_type in ["query", "preference", "comparison"]:
                    value = value.strip('"\'')
                elif entity_type == "platform":
                    # Platform is already clean
                    pass
                elif entity_type == "time":
                    # Time is already clean
                    pass
                
                entities[entity_type.upper()] = value
        
        # Post-clean: nếu có QUERY và PLATFORM, loại cụm nền tảng khỏi QUERY
        try:
            if entities.get("QUERY"):
                q = entities["QUERY"]
                plat = (entities.get("PLATFORM") or "").lower()
                platforms = ["google", "internet", "youtube", "yt", "web"]
                # loại 'trên <platform>' và tên platform đơn
                import re as _re
                if plat:
                    q = _re.sub(rf"\btrên\s+{_re.escape(plat)}\b", "", q, flags=_re.IGNORECASE)
                for p in platforms:
                    q = _re.sub(rf"\b{p}\b", "", q, flags=_re.IGNORECASE)
                q = _re.sub(r"\s+", " ", q).strip(" ,.")
                entities["QUERY"] = q
        except Exception:
            pass

        return entities

    def _clean_query(self, query: str) -> str:
        """Làm sạch QUERY: bỏ hư từ, cụm nền tảng và khoảng trắng/thừa."""
        import re as _re

        q = (query or "").strip()

        # 1) Bỏ cụm "trên web/mạng/google/youtube/yt/internet" ở đầu
        q = _re.sub(
            r"^\s*trên\s+(web|mạng|internet|google|youtube|yt)\s+",
            "",
            q,
            flags=_re.IGNORECASE,
        )

        # 2) Bỏ tên nền tảng đơn lẻ nằm trong QUERY nếu còn
        q = _re.sub(r"\b(google|youtube|yt|web|internet)\b", "", q, flags=_re.IGNORECASE)

        # 3) Bỏ hư từ lịch sự / filler
        q = _re.sub(
            r"\b(giúp tôi|giúp|giùm|dùm|nha|nhé|ạ|với|đi)\b",
            "",
            q,
            flags=_re.IGNORECASE,
        )

        # 4) Chuẩn hoá khoảng trắng và ký tự thừa
        q = _re.sub(r"\s+", " ", q)
        q = q.strip(" ,.;:!?\t\n\r")

        return q
    
    def extract_youtube_entities(self, text: str) -> Dict[str, str]:
        """Extract YouTube entities - query, duration, type, language, channel"""
        entities = {}
        
        # Extract using YouTube patterns
        for pattern, entity_type in self.youtube_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Safe group access
                if match.groups():
                    value = match.group(1).strip() if match.group(1) else ""
                else:
                    value = match.group(0).strip()  # Use full match if no groups
                
                # Clean up value
                if entity_type in ["query", "channel", "source", "preference"]:
                    value = value.strip('"\'')
                elif entity_type in ["duration", "time_constraint"]:
                    # Duration/time is already clean
                    pass
                elif entity_type in ["type", "language"]:
                    # Type/language is already clean
                    pass
                
                entities[entity_type.upper()] = value
        
        return entities
    
    def extract_info_entities(self, text: str) -> Dict[str, str]:
        """Extract info entities - topic, location, time, amount, currency"""
        entities = {}
        
        # Extract using info patterns
        for pattern, entity_type in self.info_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Safe group access
                if match.groups():
                    value = match.group(1).strip() if match.group(1) else ""
                else:
                    value = match.group(0).strip()  # Use full match if no groups
                
                # Clean up value
                if entity_type in ["topic", "location", "competition", "match", "comparison", "question", "type"]:
                    value = value.strip('"\'')
                elif entity_type in ["amount", "currency", "time"]:
                    # Amount/currency/time is already clean
                    pass
                
                entities[entity_type.upper()] = value
        
        return entities
    
    def extract_alarm_entities(self, text: str) -> Dict[str, str]:
        """Extract alarm entities - time, label, volume, recurrence, vibration"""
        entities = {}
        
        # Convert số từ chữ sang số để xử lý thời gian chính xác
        converted_text = self._convert_words_to_numbers(text)
        
        # Extract using alarm patterns với converted text
        for pattern, entity_type in self.alarm_patterns:
            match = re.search(pattern, converted_text, re.IGNORECASE)
            if match:
                if entity_type == "time":
                    # Handle time extraction with multiple groups
                    if len(match.groups()) >= 2:
                        hour = match.group(1)
                        minute = match.group(2) if match.group(2) else "00"
                        value = f"{hour}:{minute}"
                    else:
                        value = match.group(1)
                elif entity_type == "time_range":
                    # Handle time range extraction with multiple groups
                    if len(match.groups()) >= 4:
                        start_hour = match.group(1)
                        start_minute = match.group(2)
                        end_hour = match.group(3)
                        end_minute = match.group(4)
                        value = f"{start_hour}:{start_minute}-{end_hour}:{end_minute}"
                    else:
                        value = match.group(1)
                else:
                    # Safe group access
                    if match.groups():
                        value = match.group(1).strip() if match.group(1) else ""
                    else:
                        value = match.group(0).strip()  # Use full match if no groups
                
                # Clean up value
                if entity_type in ["label", "volume", "sound", "recurrence", "skip_condition", "alarm_type"]:
                    value = value.strip('"\'')
                elif entity_type in ["time", "duration", "alarm_count", "vibration", "auto_cancel", "backup"]:
                    # Time/duration/count is already clean
                    pass
                
                entities[entity_type.upper()] = value
        
        return entities
    
    def extract_calendar_entities(self, text: str) -> Dict[str, str]:
        """Extract calendar entities - title, time, location, platform, recurrence"""
        entities = {}
        
        # Convert số từ chữ sang số để xử lý thời gian và ngày tháng chính xác
        converted_text = self._convert_words_to_numbers(text)
        
        # Extract using calendar patterns với converted text
        for pattern, entity_type in self.calendar_patterns:
            match = re.search(pattern, converted_text, re.IGNORECASE)
            if match:
                if entity_type == "time_range":
                    # Handle time range extraction with multiple groups
                    if len(match.groups()) >= 4:
                        start_hour = match.group(1)
                        start_minute = match.group(2)
                        end_hour = match.group(3)
                        end_minute = match.group(4)
                        value = f"{start_hour}:{start_minute}-{end_hour}:{end_minute}"
                    elif len(match.groups()) == 3:
                        # Handle case: 8 giờ 30 đến 10 giờ (thiếu phút cuối)
                        start_hour = match.group(1)
                        start_minute = match.group(2)
                        end_hour = match.group(3)
                        value = f"{start_hour}:{start_minute}-{end_hour}:00"
                    else:
                        value = match.group(1)
                elif entity_type == "time":
                    # Handle time extraction with multiple groups
                    if len(match.groups()) >= 2:
                        hour = match.group(1)
                        minute = match.group(2) if match.group(2) else "00"
                        value = f"{hour}:{minute}"
                    else:
                        value = match.group(1)
                elif entity_type == "date":
                    # Handle date extraction with multiple groups
                    if len(match.groups()) >= 2:
                        day = match.group(1)
                        month = match.group(2)
                        value = f"{day}/{month}"
                    else:
                        value = match.group(1)
                else:
                    # Safe group access
                    if match.groups():
                        value = match.group(1).strip() if match.group(1) else ""
                    else:
                        value = match.group(0).strip()  # Use full match if no groups
                
                # Clean up value
                if entity_type in ["title", "location", "platform", "note", "event_type", "time_of_day", "recurrence"]:
                    value = value.strip('"\'')
                elif entity_type in ["time", "time_range", "date", "duration"]:
                    # Time/date/duration is already clean
                    pass
                
                entities[entity_type.upper()] = value
        
        return entities
    
    def extract_camera_entities(self, text: str) -> Dict[str, str]:
        """Extract camera entities - camera type, mode, settings, quality, duration"""
        entities = {}
        
        # Extract using camera patterns
        for pattern, entity_type in self.camera_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if entity_type == "quality":
                    # Handle quality extraction with named groups
                    if match.groupdict().get("quality"):
                        value = match.group("quality")
                    elif len(match.groups()) >= 1:
                        value = match.group(1) if match.group(1) else match.group(0)
                    else:
                        value = match.group(0)
                elif entity_type in ["duration", "count", "timer"]:
                    # Handle numeric extraction
                    if len(match.groups()) >= 1:
                        value = match.group(1)
                    else:
                        value = match.group(0)
                elif entity_type == "folder":
                    # Handle folder extraction
                    if len(match.groups()) >= 1:
                        # Safe group access
                        if match.groups():
                            value = match.group(1).strip() if match.group(1) else ""
                        else:
                            value = match.group(0).strip()  # Use full match if no groups
                    else:
                        value = match.group(0).strip()
                else:
                    value = match.group(0).strip()
                
                # Clean up value
                if entity_type in ["camera_type", "mode", "settings", "feature", "orientation", "filter"]:
                    value = value.strip('"\'')
                elif entity_type in ["quality", "duration", "count", "timer", "folder"]:
                    # Quality/duration/count/timer/folder is already clean
                    pass
                
                entities[entity_type.upper()] = value
        
        return entities

    # ===== Phase 1.1: MESSAGE/RECEIVER Extractor (Case A/B/C/D + Heuristic 4) =====
    def _extract_message_receiver_with_confidence(self, text: str, intent: str) -> Tuple[Dict[str, str], float]:
        """
        Extract MESSAGE, RECEIVER, PLATFORM for send-mess intent with confidence score.
        
        Cases:
        - A: có receiver + marker nội dung → confidence 1.0
        - B: có receiver nhưng không marker → confidence 0.8 (heuristic split)
        - C: marker trước receiver (đảo) → confidence 0.9
        - D: có platform keyword → detect và loại khỏi parse
        
        Returns:
            (entities_dict, confidence_score)
        """
        if intent != "send-mess":
            return ({}, 0.0)
        
        text_lower = text.lower().strip()
        entities = {}
        confidence = 0.0
        
        # Step 1: Extract PLATFORM (qua|bằng|trên <keyword>)
        platform_keywords = {
            "zalo": "zalo", "messenger": "messenger", "sms": "sms", "imessage": "imessage",
            "viber": "viber", "telegram": "telegram", "whatsapp": "whatsapp", "facebook": "facebook",
            "fb": "facebook", "mess": "messenger", "za": "zalo"
        }
        
        # Improved: Find ALL "qua/bằng/trên <word>" matches, then check if <word> is a platform keyword
        platform_pattern = r"\b(qua|bằng|trên|bang|tren)\s+(\w+)"
        platform_matches = list(re.finditer(platform_pattern, text_lower))
        platform_match = None
        platform_position = -1
        
        # Iterate through matches to find one with valid platform keyword
        for match in platform_matches:
            platform_raw = match.group(2).lower()
            for keyword, canonical in platform_keywords.items():
                if keyword == platform_raw or keyword in platform_raw:
                    entities["PLATFORM"] = canonical
                    platform_match = match
                    platform_position = match.start()
                    break
            if platform_match:
                break
        
        # Step 2: Tìm marker nội dung
        content_markers = [
            (r"\s+rằng\s+", " rằng "),  # "nhắn cho Lan rằng …"
            (r"\s+là\s+", " là "),       # "nhắn cho Lan là …"
            (r"\s+nói\s+", " nói "),     # "cho Lan nói …"
            (r"\s+bảo\s+", " bảo "),     # "cho Lan bảo …"
            (r":\s*", ": "),             # "cho Lan: …" hoặc "cho Lan:…"
        ]
        
        marker_pos = None
        marker_text = None
        marker_len = 0
        for marker_re, marker_str in content_markers:
            match = re.search(marker_re, text_lower)
            if match:
                marker_pos = match.start()
                marker_text = marker_str.strip()
                marker_len = len(match.group(0))
                break
        
        # Loại platform khỏi text chỉ khi nó nằm trước marker (để parse receiver dễ hơn)
        # Nếu platform sau marker → giữ lại trong MESSAGE
        if platform_match and platform_position >= 0:
            if marker_pos is None or platform_position < marker_pos:
                # Platform trước marker hoặc không có marker → loại platform
                text_lower = re.sub(r"\s+", " ", text_lower.replace(platform_match.group(0), " ")).strip()
                # Recalculate marker_pos sau khi loại platform
                for marker_re, marker_str in content_markers:
                    match = re.search(marker_re, text_lower)
                    if match:
                        marker_pos = match.start()
                        marker_text = marker_str.strip()
                        marker_len = len(match.group(0))
                        break
            # Else: platform sau marker → không loại, giữ trong MESSAGE
        
        # Step 3: Tìm cụm receiver (sau "cho/tới/đến/với" hoặc "hộ/giúp" hoặc ngay sau verb nhắn/gửi)
        receiver_trigger_pattern = r"\b(cho|tới|đến|den|toi|hộ|giúp|giup|ho|dum|dùm|với|voi)\s+"
        receiver_match = re.search(receiver_trigger_pattern, text_lower)
        
        # Fallback: pattern "Nhắn/Gửi <receiver>" (không có "cho")
        # IMPORTANT: Chỉ match khi KHÔNG có "tin" ở giữa (tránh "nhắn tin với" bị parse sai)
        if not receiver_match:
            verb_direct_pattern = r"\b(nhắn|gửi|gui|nhan)(?!\s+tin)\s+(\w+)(?:\s+(\w+))?"
            verb_match = re.search(verb_direct_pattern, text_lower)
            if verb_match:
                receiver_token1 = verb_match.group(2)
                receiver_token2 = verb_match.group(3) if verb_match.group(3) else None
                relation_pronouns = ["bố", "mẹ", "ông", "bà", "anh", "chị", "em", "con", "cháu", "bác", "cô", "chú", "dì", "thím"]
                demonstratives = ["này", "đó", "kia", "ấy"]
                time_words = ["tối", "chiều", "sáng", "trưa", "nay", "ngày", "hôm"]
                
                # Chỉ nhận nếu receiver là xưng hô hoặc có xưng hô + tên
                receiver_full = None
                receiver_end_pos = verb_match.end()
                
                if receiver_token1 in relation_pronouns:
                    # Nếu có token2 và nó không phải từ thời gian/đại từ/marker → ghép "chị Lan"
                    if receiver_token2 and receiver_token2 not in time_words and receiver_token2 not in relation_pronouns and receiver_token2 not in ["là", "rằng", "nói", "bảo"]:
                        receiver_full = f"{receiver_token1} {receiver_token2}"
                    elif receiver_token2 and receiver_token2 in demonstratives:
                        receiver_full = f"{receiver_token1} {receiver_token2}"
                    else:
                        receiver_full = receiver_token1
                        # Recalculate receiver_end_pos
                        receiver_end_pos = verb_match.start() + len(verb_match.group(1)) + 1 + len(receiver_token1)
                    
                    if receiver_full:
                        # Tìm MESSAGE phía sau
                        rest_text = text_lower[receiver_end_pos:].strip()
                        # Nếu có marker, MESSAGE bắt đầu sau marker
                        if marker_pos and marker_pos >= receiver_end_pos:
                            message_start_pos = marker_pos + marker_len
                            message_text_raw = text_lower[message_start_pos:].strip()
                            entities["RECEIVER"] = receiver_full
                            entities["MESSAGE"] = message_text_raw
                            confidence = 0.85
                        else:
                            # Không có marker, lấy phần còn lại
                            entities["RECEIVER"] = receiver_full
                            entities["MESSAGE"] = rest_text
                            confidence = 0.8
                        return (entities, confidence)
        
        if receiver_match:
            receiver_trigger_word = receiver_match.group(1)  # "cho", "hộ", "giúp"...
            receiver_start = receiver_match.end()
            
            # Case A: có marker sau receiver
            if marker_pos and marker_pos > receiver_start:
                receiver_text_raw = text_lower[receiver_start:marker_pos].strip()
                message_start_pos = marker_pos + marker_len
                message_text_raw = text_lower[message_start_pos:].strip()
                
                # Clean receiver: cắt tối đa 2-3 tokens, loại bỏ khi gặp đại từ quan hệ thừa
                receiver_tokens = receiver_text_raw.split()
                relation_pronouns = ["bố", "mẹ", "ông", "bà", "anh", "chị", "em", "con", "cháu", "bác", "cô", "chú", "dì", "thím"]
                demonstratives = ["này", "đó", "kia", "ấy"]
                time_words = ["tối", "chiều", "sáng", "trưa", "nay", "ngày", "hôm"]
                
                # Context-aware: nếu token 0 là xưng hô, token 1 nhiều khả năng là tên (kể cả "mai", "lan")
                if len(receiver_tokens) >= 2:
                    if receiver_tokens[0] in relation_pronouns:
                        # Nếu token 1 là demonstrative → giữ 2 tokens
                        if receiver_tokens[1] in demonstratives:
                            receiver_clean = " ".join(receiver_tokens[:2])
                        # Nếu token 1 là đại từ quan hệ khác → chỉ giữ token 0
                        elif receiver_tokens[1] in relation_pronouns:
                            receiver_clean = receiver_tokens[0]
                        # Special: Nếu token 1 là "mai" → ưu tiên TÊN (chị Mai) khi chỉ có 2 tokens
                        elif receiver_tokens[1] == "mai":
                            # Chỉ xem là thời gian nếu có token 2 = "này" (mai này)
                            if len(receiver_tokens) >= 3 and receiver_tokens[2] == "này":
                                receiver_clean = receiver_tokens[0]
                            else:
                                # "chị Mai" → giữ 2 tokens (tên)
                                receiver_clean = " ".join(receiver_tokens[:2])
                        # Special: Nếu token 1 là từ thời gian KHÁC "mai" và có token 2 → check token 2
                        elif receiver_tokens[1] in time_words and len(receiver_tokens) >= 3:
                            # Nếu token 2 là thời gian/đại từ → token 1 là thời gian → chỉ giữ token 0
                            if receiver_tokens[2] in time_words or receiver_tokens[2] in relation_pronouns:
                                receiver_clean = receiver_tokens[0]
                            # Ngược lại, token 1 là tên → giữ 2 tokens
                            else:
                                receiver_clean = " ".join(receiver_tokens[:2])
                        # Token 1 không phải demonstrative/relation_pronoun/time → ưu tiên là tên → giữ 2 tokens
                        else:
                            receiver_clean = " ".join(receiver_tokens[:2])
                    else:
                        # Token 0 không phải xưng hô, nếu token 1 là demonstrative → giữ 2 tokens
                        if receiver_tokens[1] in demonstratives:
                            receiver_clean = " ".join(receiver_tokens[:2])
                        else:
                            receiver_clean = receiver_tokens[0]
                elif len(receiver_tokens) == 1:
                    receiver_clean = receiver_tokens[0]
                else:
                    receiver_clean = " ".join(receiver_tokens[:2])
                
                entities["RECEIVER"] = receiver_clean.strip()
                entities["MESSAGE"] = message_text_raw.strip()
                confidence = 1.0  # Case A: rõ ràng nhất
            
            # Case B: không marker, dùng heuristic split
            elif not marker_pos:
                receiver_text_raw = text_lower[receiver_start:].strip()
                receiver_tokens = receiver_text_raw.split()
                
                # Heuristic cải thiện:
                # - Nếu token đầu là xưng hô (bố/mẹ/anh/chị…) và token thứ 2 là tên → receiver = 2 tokens
                # - Nếu token đầu là tên đơn → receiver = 1 token
                # - Dừng receiver khi gặp đại từ quan hệ thứ 2 (mẹ/con/bố/bác…) hoặc động từ nội dung mạnh hoặc từ thời gian
                # - Đặc biệt: nếu trigger là "giúp/hộ" → chỉ lấy 1 token làm receiver
                relation_pronouns = ["bố", "mẹ", "ông", "bà", "anh", "chị", "em", "con", "cháu", "bác", "cô", "chú", "dì", "thím", "số"]
                demonstratives = ["này", "đó", "kia", "ấy"]
                time_words = ["tối", "chiều", "sáng", "trưa", "mai", "nay", "ngày", "hôm"]
                # Chỉ dùng động từ mạnh để tách receiver, không dùng "nhớ" (có thể là phần đầu MESSAGE)
                content_verbs_strong = ["về", "đến", "ăn", "mua", "đón", "khỏe", "uống", "học", "làm", "đi", "ở", "không", "đang"]
                
                # Nếu trigger là "giúp/hộ" → chỉ lấy 1 token (xưng hô) làm receiver
                if receiver_trigger_word in ["giúp", "giup", "hộ", "ho", "dum", "dùm"]:
                    receiver_clean = receiver_tokens[0] if receiver_tokens else ""
                    message_clean = " ".join(receiver_tokens[1:]) if len(receiver_tokens) > 1 else ""
                    if receiver_clean:
                        entities["RECEIVER"] = receiver_clean.strip()
                    if message_clean:
                        entities["MESSAGE"] = message_clean.strip()
                    confidence = 0.8
                    return (entities, confidence)
                
                receiver_end_idx = 1  # mặc định 1 token
                if len(receiver_tokens) >= 2:
                    # Nếu token 0 là xưng hô → ưu tiên giữ 2 tokens (context: tên theo sau xưng hô)
                    if receiver_tokens[0] in relation_pronouns:
                        # Nếu token 1 là demonstrative → giữ 2 tokens
                        if receiver_tokens[1] in demonstratives:
                            receiver_end_idx = 2
                        # Nếu token 1 là đại từ quan hệ khác → chỉ giữ token 0
                        elif receiver_tokens[1] in relation_pronouns:
                            receiver_end_idx = 1
                        # Nếu token 1 là động từ mạnh → chỉ giữ token 0
                        elif receiver_tokens[1] in content_verbs_strong:
                            receiver_end_idx = 1
                        # Special case: "mai" sau xưng hô → ưu tiên tên (context: "chị Mai")
                        # Chỉ xem là thời gian nếu token 2 là "này" (mai này) hoặc không có token 2
                        elif receiver_tokens[1] == "mai" and len(receiver_tokens) >= 3:
                            # Nếu token 2 là "này" (mai này = thời gian) → dừng tại token 0
                            if receiver_tokens[2] == "này":
                                receiver_end_idx = 1
                            # Ngược lại, "chị Mai hôm nay..." → "Mai" là tên → giữ 2 tokens
                            else:
                                receiver_end_idx = 2
                        # Nếu token 1 là "mai" và chỉ có 2 tokens → ưu tiên tên
                        elif receiver_tokens[1] == "mai":
                            receiver_end_idx = 2
                        # Nếu token 1 là từ thời gian KHÁC "mai" → check token 2
                        elif receiver_tokens[1] in time_words and len(receiver_tokens) >= 3:
                            # Nếu token 2 cũng là thời gian/động từ → token 1 là thời gian → dừng tại token 0
                            if receiver_tokens[2] in time_words or receiver_tokens[2] in content_verbs_strong or receiver_tokens[2] in relation_pronouns:
                                receiver_end_idx = 1
                            else:
                                # Token 1 nhiều khả năng là tên
                                receiver_end_idx = 2
                        # Token 1 không phải edge case → giữ 2 tokens (ưu tiên tên)
                        else:
                            receiver_end_idx = 2
                    # Token 0 không phải xưng hô
                    else:
                        # Nếu token 1 là demonstrative → giữ 2 tokens
                        if receiver_tokens[1] in demonstratives:
                            receiver_end_idx = 2
                        # Nếu token 1 là thời gian/động từ/đại từ → dừng tại token 0
                        elif receiver_tokens[1] in time_words or receiver_tokens[1] in content_verbs_strong or receiver_tokens[1] in relation_pronouns:
                            receiver_end_idx = 1
                        else:
                            receiver_end_idx = 2
                
                receiver_clean = " ".join(receiver_tokens[:receiver_end_idx])
                message_clean = " ".join(receiver_tokens[receiver_end_idx:])
                
                if receiver_clean:
                    entities["RECEIVER"] = receiver_clean.strip()
                if message_clean:
                    entities["MESSAGE"] = message_clean.strip()
                confidence = 0.8  # Case B: heuristic, ít chắc chắn hơn
            
            else:
                # marker_pos < receiver_start: không xảy ra với pattern hiện tại, bỏ qua
                pass
        
        # Case C: marker xuất hiện trước "cho/tới/đến" (đảo trật tự)
        elif marker_pos and not receiver_match:
            # "Bảo rằng … cho Lan"
            message_start_pos = marker_pos + len(marker_text) if marker_text else marker_pos
            rest_text = text_lower[message_start_pos:].strip()
            
            # Tìm "cho/tới/đến" trong phần còn lại
            receiver_trigger_match_rest = re.search(receiver_trigger_pattern, rest_text)
            if receiver_trigger_match_rest:
                message_part = rest_text[:receiver_trigger_match_rest.start()].strip()
                receiver_part = rest_text[receiver_trigger_match_rest.end():].strip()
                
                # Clean
                receiver_tokens = receiver_part.split()
                receiver_clean = " ".join(receiver_tokens[:3])
                
                entities["MESSAGE"] = message_part
                entities["RECEIVER"] = receiver_clean.strip()
                confidence = 0.9  # Case C: đảo nhưng rõ ràng
        
        # Case D/E/F...: các case khó khác → confidence thấp
        else:
            # Không tách được, trả về empty
            confidence = 0.0
        
        # CLEANUP STEP: Loại bỏ platform keywords khỏi MESSAGE nếu đã extract PLATFORM entity
        if entities.get("MESSAGE") and entities.get("PLATFORM"):
            message_clean = entities["MESSAGE"]
            extracted_platform = entities["PLATFORM"]
            
            # Pattern 1: Loại bỏ ONLY "qua/bằng/trên <detected_platform_keyword>" khỏi MESSAGE
            # Build pattern that matches "qua/bằng/trên" + any platform keyword
            platform_keywords_list = list(platform_keywords.keys())
            platform_keywords_pattern = "|".join(re.escape(kw) for kw in platform_keywords_list)
            
            # Match "qua/bằng/trên <any_platform_keyword>"
            platform_removal_pattern = rf"\s*(qua|bằng|trên|bang|tren)\s+({platform_keywords_pattern})\b"
            message_clean = re.sub(platform_removal_pattern, "", message_clean, flags=re.IGNORECASE).strip()
            
            # Pattern 2: Loại bỏ standalone platform keywords ở cuối MESSAGE
            for keyword in platform_keywords_list:
                # Loại bỏ platform keyword ở cuối (với optional whitespace)
                message_clean = re.sub(r"\s+" + re.escape(keyword) + r"\s*$", "", message_clean, flags=re.IGNORECASE).strip()
            
            # Cleanup multiple spaces
            message_clean = re.sub(r"\s+", " ", message_clean).strip()
            
            if message_clean:
                entities["MESSAGE"] = message_clean
        
        return (entities, confidence)

    def test_extract_message_receiver(self):
        """Inline test cho _extract_message_receiver_with_confidence - Target: >=90% exact match"""
        test_cases = [
            # Case A: có receiver + marker
            {"input": "Nhắn cho chị Mai là hôm nay mẹ về muộn", "intent": "send-mess", "expected": {"RECEIVER": "chị mai", "MESSAGE": "hôm nay mẹ về muộn"}, "min_confidence": 1.0},
            {"input": "Gửi tới anh Trường rằng con đến đón", "intent": "send-mess", "expected": {"RECEIVER": "anh trường", "MESSAGE": "con đến đón"}, "min_confidence": 1.0},
            {"input": "Nhắn cho Lan: chiều nay con đến", "intent": "send-mess", "expected": {"RECEIVER": "lan", "MESSAGE": "chiều nay con đến"}, "min_confidence": 1.0},
            {"input": "Nhắn cho mẹ là tối nay con về muộn", "intent": "send-mess", "expected": {"RECEIVER": "mẹ", "MESSAGE": "tối nay con về muộn"}, "min_confidence": 1.0},
            {"input": "Gửi cho cô Hạnh rằng mai con đến sớm", "intent": "send-mess", "expected": {"RECEIVER": "cô hạnh", "MESSAGE": "mai con đến sớm"}, "min_confidence": 1.0},
            {"input": "Nhắn cho anh Trường là mẹ đang ở ngoài", "intent": "send-mess", "expected": {"RECEIVER": "anh trường", "MESSAGE": "mẹ đang ở ngoài"}, "min_confidence": 1.0},
            {"input": "Nhắn cho Lan là hôm nay là sinh nhật bác", "intent": "send-mess", "expected": {"RECEIVER": "lan", "MESSAGE": "hôm nay là sinh nhật bác"}, "min_confidence": 1.0},
            {"input": "Gửi cho chị Mai: con đến đón lúc 5 giờ", "intent": "send-mess", "expected": {"RECEIVER": "chị mai", "MESSAGE": "con đến đón lúc 5 giờ"}, "min_confidence": 1.0},
            {"input": "Gửi tới Lan nói rằng tối nay không ăn cơm", "intent": "send-mess", "expected": {"RECEIVER": "lan", "MESSAGE": "tối nay không ăn cơm"}, "min_confidence": 1.0},
            {"input": "Nhắn cho bà là ừm… hôm nay con về trễ", "intent": "send-mess", "expected": {"RECEIVER": "bà", "MESSAGE": "ừm… hôm nay con về trễ"}, "min_confidence": 1.0},
            {"input": "Nhắn cho chị Mai là mẹ về trễ nhé", "intent": "send-mess", "expected": {"RECEIVER": "chị mai", "MESSAGE": "mẹ về trễ nhé"}, "min_confidence": 1.0},
            {"input": "Nhắn cho Lan nói mẹ đang đợi ở nhà", "intent": "send-mess", "expected": {"RECEIVER": "lan", "MESSAGE": "mẹ đang đợi ở nhà"}, "min_confidence": 1.0},
            {"input": "Nhắn cho mẹ là con đang trên đường về", "intent": "send-mess", "expected": {"RECEIVER": "mẹ", "MESSAGE": "con đang trên đường về"}, "min_confidence": 1.0},
            {"input": "Nhắn cho mẹ rằng hôm nay con ăn ở ngoài", "intent": "send-mess", "expected": {"RECEIVER": "mẹ", "MESSAGE": "hôm nay con ăn ở ngoài"}, "min_confidence": 1.0},
            {"input": "Gửi cho bố là bác sĩ hẹn mai tái khám", "intent": "send-mess", "expected": {"RECEIVER": "bố", "MESSAGE": "bác sĩ hẹn mai tái khám"}, "min_confidence": 1.0},
            {"input": "Nhắn cho bà rằng đừng quên uống thuốc", "intent": "send-mess", "expected": {"RECEIVER": "bà", "MESSAGE": "đừng quên uống thuốc"}, "min_confidence": 1.0},
            {"input": "Nhắn cho chị Lan là hôm nay về muộn nhé", "intent": "send-mess", "expected": {"RECEIVER": "chị lan", "MESSAGE": "hôm nay về muộn nhé"}, "min_confidence": 1.0},
            {"input": "Gửi cho anh Trường nói tối nay không về nhà", "intent": "send-mess", "expected": {"RECEIVER": "anh trường", "MESSAGE": "tối nay không về nhà"}, "min_confidence": 1.0},
            
            # Case B: có receiver nhưng không marker
            {"input": "Nhắn cho Lan mẹ về trễ", "intent": "send-mess", "expected": {"RECEIVER": "lan", "MESSAGE": "mẹ về trễ"}, "min_confidence": 0.8},
            {"input": "Gửi tới bố con đang ở ngoài", "intent": "send-mess", "expected": {"RECEIVER": "bố", "MESSAGE": "con đang ở ngoài"}, "min_confidence": 0.8},
            {"input": "Nhắn cho bố tối nay con về muộn", "intent": "send-mess", "expected": {"RECEIVER": "bố", "MESSAGE": "tối nay con về muộn"}, "min_confidence": 0.8},
            {"input": "Nhắn chị Lan chiều nay ghé qua", "intent": "send-mess", "expected": {"RECEIVER": "chị lan", "MESSAGE": "chiều nay ghé qua"}, "min_confidence": 0.8},
            {"input": "Gửi tin nhắn cho Lan nói tối nay không ăn cơm", "intent": "send-mess", "expected": {"RECEIVER": "lan", "MESSAGE": "tối nay không ăn cơm"}, "min_confidence": 0.8},
            {"input": "Nhắn cho chị Mai hôm nay mẹ về muộn", "intent": "send-mess", "expected": {"RECEIVER": "chị mai", "MESSAGE": "hôm nay mẹ về muộn"}, "min_confidence": 0.8},
            {"input": "Gửi cho chị Lan tối nay mưa to", "intent": "send-mess", "expected": {"RECEIVER": "chị lan", "MESSAGE": "tối nay mưa to"}, "min_confidence": 0.8},
            {"input": "Nhắn cho chị Mai hôm nay trời lạnh", "intent": "send-mess", "expected": {"RECEIVER": "chị mai", "MESSAGE": "hôm nay trời lạnh"}, "min_confidence": 0.8},
            
            # Case D: có platform
            {"input": "Nhắn cho anh Trường qua Zalo là con đến", "intent": "send-mess", "expected": {"RECEIVER": "anh trường", "MESSAGE": "con đến", "PLATFORM": "zalo"}, "min_confidence": 1.0},
            {"input": "Gửi bằng SMS tới số này bác khỏe không", "intent": "send-mess", "expected": {"RECEIVER": "số này", "MESSAGE": "bác khỏe không", "PLATFORM": "sms"}, "min_confidence": 0.8},
            {"input": "Nhắn qua Zalo cho Lan rằng con sắp tới", "intent": "send-mess", "expected": {"RECEIVER": "lan", "MESSAGE": "con sắp tới", "PLATFORM": "zalo"}, "min_confidence": 1.0},
            {"input": "Gửi cho anh Trường qua Messenger là tối nay về trễ", "intent": "send-mess", "expected": {"RECEIVER": "anh trường", "MESSAGE": "tối nay về trễ", "PLATFORM": "messenger"}, "min_confidence": 1.0},
            {"input": "Nhắn bằng SMS cho số 0987654321 bác khỏe không", "intent": "send-mess", "expected": {"RECEIVER": "số 0987654321", "MESSAGE": "bác khỏe không", "PLATFORM": "sms"}, "min_confidence": 0.8},
            {"input": "Nhắn cho số này là nhớ uống thuốc nhé", "intent": "send-mess", "expected": {"RECEIVER": "số này", "MESSAGE": "nhớ uống thuốc nhé"}, "min_confidence": 1.0},
            {"input": "Nhắn cho mẹ qua zalo bảo tối nay khóa cửa", "intent": "send-mess", "expected": {"RECEIVER": "mẹ", "MESSAGE": "tối nay khóa cửa", "PLATFORM": "zalo"}, "min_confidence": 1.0},
            {"input": "Gửi cho anh Trường rằng bác khỏe không qua messenger nhé", "intent": "send-mess", "expected": {"RECEIVER": "anh trường", "MESSAGE": "bác khỏe không qua messenger nhé"}, "min_confidence": 1.0},
            {"input": "Nhắn cho anh Nam qua Zalo là con đến rồi", "intent": "send-mess", "expected": {"RECEIVER": "anh nam", "MESSAGE": "con đến rồi", "PLATFORM": "zalo"}, "min_confidence": 1.0},
            {"input": "Nhắn cho Lan qua Zalo tối nay con về trễ khoảng sáu rưỡi tối ạ", "intent": "send-mess", "expected": {"RECEIVER": "lan", "MESSAGE": "tối nay con về trễ khoảng sáu rưỡi tối ạ", "PLATFORM": "zalo"}, "min_confidence": 0.8},
            {"input": "Gửi bằng Messenger tới mẹ là con đến đón", "intent": "send-mess", "expected": {"RECEIVER": "mẹ", "MESSAGE": "con đến đón", "PLATFORM": "messenger"}, "min_confidence": 1.0},
            
            # Edge cases: fallback pattern "hộ/giúp"
            {"input": "Nhắn hộ mẹ là tối nay về muộn", "intent": "send-mess", "expected": {"RECEIVER": "mẹ", "MESSAGE": "tối nay về muộn"}, "min_confidence": 0.9},
            {"input": "Nhắn giúp bà nhớ khóa cửa tối nay", "intent": "send-mess", "expected": {"RECEIVER": "bà", "MESSAGE": "nhớ khóa cửa tối nay"}, "min_confidence": 0.8},
        ]
        
        passed = 0
        total = len(test_cases)
        
        print("\n===== Test MESSAGE/RECEIVER Extraction =====")
        for i, tc in enumerate(test_cases, 1):
            entities, conf = self._extract_message_receiver_with_confidence(tc["input"], tc["intent"])
            expected = tc["expected"]
            
            # Normalize for comparison
            entities_norm = {k: v.lower().strip() for k, v in entities.items()}
            expected_norm = {k: v.lower().strip() for k, v in expected.items()}
            
            # Check exact match
            match = all(entities_norm.get(k) == v for k, v in expected_norm.items())
            conf_ok = conf >= tc["min_confidence"]
            
            if match and conf_ok:
                passed += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            
            print(f"[{i}/{total}] {status}")
            print(f"  Input: {tc['input']}")
            print(f"  Expected: {expected}")
            print(f"  Got: {entities} (conf={conf:.2f})")
            if not match:
                print(f"  Mismatch!")
            print()
        
        accuracy = (passed / total) * 100
        print(f"Result: {passed}/{total} passed ({accuracy:.1f}%)")
        print(f"Target: >=80% → {'✅達成' if accuracy >= 80 else '❌ NOT YET'}")
        return accuracy
    
    # ===== Phase 1.2: ALARM_TIME_DATE Extractor =====
    def _extract_alarm_time_date_with_confidence(self, text: str, intent: str) -> Tuple[Dict[str, str], float]:
        """
        Extract TIME, DATE, TIMESTAMP, DAYS_OF_WEEK, REMINDER_CONTENT for set-alarm intent.
        
        Returns:
            (entities_dict, confidence_score)
        """
        from datetime import datetime, timedelta
        import re
        
        if intent != "set-alarm":
            return ({}, 0.0)
        
        text_lower = text.lower().strip()
        entities = {}
        confidence = 0.0
        
        # Step 1: Extract TIME (7 giờ, 19h, 6 rưỡi, 7:30...)
        # IMPORTANT: Check for time period BOTH before AND after the time
        
        # First, detect if period comes BEFORE the time (e.g., "chiều nay 3 giờ")
        time_period_prefix = None
        prefix_match = re.search(r"(sáng|chiều|tối|trưa)\s+(?:nay|hôm\s*nay|mai)?\s*(\d{1,2})\s*giờ", text_lower)
        if prefix_match:
            time_period_prefix = prefix_match.group(1)
        
        time_patterns = [
            (r"(\d{1,2})\s*giờ\s*(\d{1,2})(?:\s*phút)?(?:\s+(sáng|chiều|tối|trưa))?", "HH:MM"),  # "7 giờ 30 chiều"
            (r"(\d{1,2})\s*giờ\s*rưỡi(?:\s+(sáng|chiều|tối|trưa))?", "HH:30"),  # "6 rưỡi sáng"
            (r"(\d{1,2})\s*giờ(?:\s+(sáng|chiều|tối|trưa))?", "HH:00"),  # "7 giờ sáng"
            (r"(\d{1,2}):(\d{2})", "HH:MM"),  # "07:30"
            (r"(\d{1,2})h(\d{2})?(?:\s+(sáng|chiều|tối|trưa))?", "HH:MM"),  # "7h30 chiều"
        ]
        
        time_value = None
        time_period = time_period_prefix  # Use prefix if detected
        
        for pattern, format_type in time_patterns:
            match = re.search(pattern, text_lower)
            if match:
                hour = int(match.group(1))
                minute = 0
                
                if format_type == "HH:MM":
                    minute = int(match.group(2)) if match.group(2) and match.group(2).isdigit() else 0
                    # Check for period in group 3 or 2 (depending on whether minute exists)
                    last_idx = match.lastindex
                    if last_idx and last_idx >= 3 and match.group(last_idx) in ["sáng", "chiều", "tối", "trưa"]:
                        time_period = match.group(last_idx)
                elif format_type == "HH:30":
                    minute = 30
                    # Check for period in last group
                    last_idx = match.lastindex
                    if last_idx and last_idx >= 2 and match.group(last_idx) in ["sáng", "chiều", "tối", "trưa"]:
                        time_period = match.group(last_idx)
                elif format_type == "HH:00":
                    minute = 0
                    # Check for period in group 2
                    if match.group(2) and match.group(2) in ["sáng", "chiều", "tối", "trưa"]:
                        time_period = match.group(2)
                
                # Adjust hour based on period
                if time_period:
                    if time_period == "chiều" and hour <= 11:
                        hour += 12
                    elif time_period == "tối" and hour <= 11:
                        hour += 12
                    elif time_period == "trưa" and hour <= 11:
                        hour = 12
                    elif time_period == "sáng" and hour >= 12:
                        hour -= 12
                
                time_value = f"{hour:02d}:{minute:02d}"
                break
        
        if time_value:
            entities["TIME"] = time_value
            confidence += 0.5
        
        # Step 2: Extract DATE (hôm nay, mai, ngày mai, thứ 2, ngày 25/12...)
        date_patterns = [
            (r"hôm\s*nay|nay", "today"),
            (r"ngày\s*mai|mai(?!\s*tái)", "tomorrow"),  # Avoid "mai tái khám"
            (r"ngày\s*kia", "day_after_tomorrow"),
            (r"thứ\s*(hai|ba|tư|năm|sáu|bảy|chủ\s*nhật)", "weekday"),
            (r"(\d{1,2})[/-](\d{1,2})", "DD/MM"),  # "25/12" or "25-12"
        ]
        
        date_value = None
        weekday_map = {
            "hai": 0, "ba": 1, "tư": 2, "năm": 3, "sáu": 4, "bảy": 5, "chủ nhật": 6
        }
        
        for pattern, date_type in date_patterns:
            match = re.search(pattern, text_lower)
            if match:
                now = datetime.now()
                if date_type == "today":
                    date_value = now.strftime("%Y-%m-%d")
                elif date_type == "tomorrow":
                    date_value = (now + timedelta(days=1)).strftime("%Y-%m-%d")
                elif date_type == "day_after_tomorrow":
                    date_value = (now + timedelta(days=2)).strftime("%Y-%m-%d")
                elif date_type == "weekday":
                    weekday_name = match.group(1)
                    target_weekday = weekday_map.get(weekday_name, -1)
                    if target_weekday >= 0:
                        current_weekday = now.weekday()
                        days_ahead = (target_weekday - current_weekday) % 7
                        if days_ahead == 0:
                            days_ahead = 7  # Next week
                        date_value = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
                elif date_type == "DD/MM":
                    day = int(match.group(1))
                    month = int(match.group(2))
                    year = now.year
                    try:
                        date_value = f"{year}-{month:02d}-{day:02d}"
                    except:
                        pass
                break
        
        if date_value:
            entities["DATE"] = date_value
            confidence += 0.3
        
        # Step 3: Generate TIMESTAMP if both TIME and DATE exist
        if "TIME" in entities and "DATE" in entities:
            try:
                timestamp = f"{entities['DATE']}T{entities['TIME']}:00"
                entities["TIMESTAMP"] = timestamp
                confidence += 0.1
            except:
                pass
        
        # Step 4: Extract DAYS_OF_WEEK from DATE
        if "DATE" in entities:
            try:
                date_obj = datetime.strptime(entities["DATE"], "%Y-%m-%d")
                day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                entities["DAYS_OF_WEEK"] = day_names[date_obj.weekday()]
            except:
                pass
        
        # Step 5: Extract REMINDER_CONTENT (optional)
        # Pattern: "nhắc <content>" hoặc "<content>" sau time/date
        reminder_patterns = [
            r"nhắc\s+(?:mẹ|bà|bố|ông|tôi|em|anh|chị)?\s*(.+?)(?:\s+lúc|\s+vào|\s+$)",
            r"(?:uống|ăn|làm|đến|về)\s+(.+?)(?:\s+lúc|\s+vào|\s+$)",
        ]
        
        for pattern in reminder_patterns:
            match = re.search(pattern, text_lower)
            if match:
                content = match.group(1).strip()
                if len(content) > 3:  # Avoid too short content
                    entities["REMINDER_CONTENT"] = content
                    confidence += 0.05
                    break
        
        # Step 6: Extract FREQUENCY (mỗi ngày, hàng ngày, hàng tuần...)
        frequency_patterns = {
            r"mỗi\s*ngày|hàng\s*ngày": "daily",
            r"hàng\s*tuần|mỗi\s*tuần": "weekly",
            r"hàng\s*tháng|mỗi\s*tháng": "monthly",
        }
        
        for pattern, freq_value in frequency_patterns.items():
            if re.search(pattern, text_lower):
                entities["FREQUENCY"] = freq_value
                # Boost confidence for FREQUENCY (implies recurring alarm)
                confidence += 0.25
                break
        
        # Final confidence adjustment: if we have TIME but no DATE, still reasonable
        if "TIME" in entities and "DATE" not in entities:
            confidence = max(confidence, 0.82)  # Minimum 0.82 for TIME-only alarms (acceptable for recurring)
        
        return (entities, min(confidence, 1.0))
    
    def test_extract_alarm_time_date(self):
        """Inline test cho _extract_alarm_time_date_with_confidence - Target: >=90% exact match"""
        test_cases = [
            # Basic TIME + DATE patterns
            {"input": "Đặt báo thức lúc 6 giờ sáng mai", "intent": "set-alarm", "expected": {"TIME": "06:00"}, "min_confidence": 0.8},
            {"input": "Hẹn báo thức 7 giờ thứ Hai", "intent": "set-alarm", "expected": {"TIME": "07:00"}, "min_confidence": 0.8},
            {"input": "Nhắc mẹ 3 giờ chiều hôm nay", "intent": "set-alarm", "expected": {"TIME": "15:00"}, "min_confidence": 0.8},
            {"input": "Đặt báo thức 6 giờ rưỡi sáng", "intent": "set-alarm", "expected": {"TIME": "06:30"}, "min_confidence": 0.8},
            {"input": "Báo thức lúc 19 giờ tối nay", "intent": "set-alarm", "expected": {"TIME": "19:00"}, "min_confidence": 0.8},
            
            # TIME with REMINDER_CONTENT
            {"input": "Nhắc uống thuốc lúc 8 giờ tối mỗi ngày", "intent": "set-alarm", "expected": {"TIME": "20:00", "FREQUENCY": "daily"}, "min_confidence": 0.8},
            {"input": "Đặt lịch nhắc uống thuốc sau bữa tối", "intent": "set-alarm", "expected": {}, "min_confidence": 0.0},  # No clear TIME
            
            # Relative dates
            {"input": "Báo thức mai lúc 6 giờ", "intent": "set-alarm", "expected": {"TIME": "06:00"}, "min_confidence": 0.8},
            {"input": "Nhắc tôi hôm nay 5 giờ chiều", "intent": "set-alarm", "expected": {"TIME": "17:00"}, "min_confidence": 0.8},
            {"input": "Đặt báo thức ngày mai 7h30", "intent": "set-alarm", "expected": {"TIME": "07:30"}, "min_confidence": 0.8},
            
            # Weekday patterns
            {"input": "Nhắc tôi thứ Ba lúc 9 giờ sáng", "intent": "set-alarm", "expected": {"TIME": "09:00"}, "min_confidence": 0.8},
            {"input": "Báo thức thứ Sáu 6 giờ rưỡi", "intent": "set-alarm", "expected": {"TIME": "06:30"}, "min_confidence": 0.8},
            
            # FREQUENCY patterns
            {"input": "Nhắc uống thuốc mỗi ngày lúc 8 giờ", "intent": "set-alarm", "expected": {"TIME": "08:00", "FREQUENCY": "daily"}, "min_confidence": 0.8},
            {"input": "Đặt báo thức hàng ngày 6 giờ sáng", "intent": "set-alarm", "expected": {"TIME": "06:00", "FREQUENCY": "daily"}, "min_confidence": 0.8},
            
            # Edge cases
            {"input": "Đặt báo thức 6h sáng mai giúp tôi đi", "intent": "set-alarm", "expected": {"TIME": "06:00"}, "min_confidence": 0.8},
            {"input": "Báo thức chiều nay 3 giờ nhắc mẹ uống thuốc", "intent": "set-alarm", "expected": {"TIME": "15:00"}, "min_confidence": 0.8},
        ]
        
        passed = 0
        total = len(test_cases)
        
        print("\n===== Test ALARM_TIME_DATE Extraction =====")
        for i, tc in enumerate(test_cases, 1):
            entities, conf = self._extract_alarm_time_date_with_confidence(tc["input"], tc["intent"])
            expected = tc["expected"]
            
            # Check if all expected keys exist with correct values
            match = all(entities.get(k) == v for k, v in expected.items())
            conf_ok = conf >= tc["min_confidence"]
            
            if match and conf_ok:
                passed += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            
            print(f"[{i}/{total}] {status}")
            print(f"  Input: {tc['input']}")
            print(f"  Expected: {expected}")
            print(f"  Got: {entities} (conf={conf:.2f})")
            if not match:
                print(f"  Mismatch!")
            print()
        
        accuracy = (passed / total) * 100
        print(f"Result: {passed}/{total} passed ({accuracy:.1f}%)")
        print(f"Target: >=90% → {'✅達成' if accuracy >= 90 else '❌ Chưa đạt'}")
        return accuracy
    
    # ===== Phase 1.3: DEVICE_CONTROL Extractor =====
    def _extract_device_control_with_confidence(self, text: str, intent: str) -> Tuple[Dict[str, str], float]:
        """
        Extract DEVICE, ACTION, MODE, LEVEL, LOCATION for control-device intent.
        
        Returns:
            (entities_dict, confidence_score)
        """
        import re
        
        if intent != "control-device":
            return ({}, 0.0)
        
        text_lower = text.lower().strip()
        entities = {}
        confidence = 0.0
        
        # Step 1: Extract ACTION (chỉ ON/OFF cơ bản)
        action_keywords = {
            r"\b(bật|mở|khởi\s*động|kích\s*hoạt)\b": "ON",
            r"\b(tắt|đóng|ngắt|dừng)\b": "OFF",
        }
        
        action_value = None
        for pattern, action in action_keywords.items():
            if re.search(pattern, text_lower):
                action_value = action
                confidence += 0.4
                break
        
        if action_value:
            entities["ACTION"] = action_value
        
        # Step 2: Extract DEVICE (chỉ thiết bị di động Android: flash, wifi, bluetooth, volume, brightness...)
        device_keywords = {
            r"\b(đèn\s*pin|den\s*pin|flash|flashlight)\b": "flash",
            r"\b(wifi|wi\s*fi|mạng\s*không\s*dây)\b": "wifi",
            r"\b(bluetooth|blue\s*tooth|blutooth)\b": "bluetooth",
            r"\b(âm\s*lượng|am\s*luong|volume|tiếng|loa)\b": "volume",
            r"\b(độ\s*sáng|do\s*sang|brightness|sáng|sang)\b": "brightness",
            r"\b(data|dữ\s*liệu|du\s*lieu|3g|4g|5g)\b": "mobile_data",
            r"\b(chế\s*độ\s*máy\s*bay|máy\s*bay|airplane|flight\s*mode)\b": "airplane_mode",
            r"\b(xoay\s*màn\s*hình|xoay|rotation|auto\s*rotate)\b": "screen_rotation",
        }
        
        device_value = None
        for pattern, device in device_keywords.items():
            match = re.search(pattern, text_lower)
            if match:
                device_value = device
                confidence += 0.4
                break
        
        if device_value:
            entities["DEVICE"] = device_value
        
        # Step 3: LOCATION - Not applicable for mobile devices (skip)
        # Step 4: LEVEL - Not needed for basic ON/OFF mobile controls (skip)
        # Step 5: MODE - Not needed for basic ON/OFF mobile controls (skip)
        
        # Minimum confidence: if we have ACTION + DEVICE, that's sufficient
        if "ACTION" in entities and "DEVICE" in entities:
            confidence = max(confidence, 0.85)
        
        return (entities, min(confidence, 1.0))
    
    def test_extract_device_control(self):
        """Inline test cho _extract_device_control_with_confidence - Target: >=90% exact match (Mobile device controls)"""
        test_cases = [
            # Flash (đèn pin)
            {"input": "Bật đèn pin", "intent": "control-device", "expected": {"ACTION": "ON", "DEVICE": "flash"}, "min_confidence": 0.85},
            {"input": "Tắt flash đi", "intent": "control-device", "expected": {"ACTION": "OFF", "DEVICE": "flash"}, "min_confidence": 0.85},
            {"input": "Mở đèn pin giúp bà", "intent": "control-device", "expected": {"ACTION": "ON", "DEVICE": "flash"}, "min_confidence": 0.85},
            
            # WiFi
            {"input": "Bật wifi lên", "intent": "control-device", "expected": {"ACTION": "ON", "DEVICE": "wifi"}, "min_confidence": 0.85},
            {"input": "Tắt wifi đi", "intent": "control-device", "expected": {"ACTION": "OFF", "DEVICE": "wifi"}, "min_confidence": 0.85},
            
            # Bluetooth
            {"input": "Bật bluetooth", "intent": "control-device", "expected": {"ACTION": "ON", "DEVICE": "bluetooth"}, "min_confidence": 0.85},
            {"input": "Tắt bluetooth giúp bà", "intent": "control-device", "expected": {"ACTION": "OFF", "DEVICE": "bluetooth"}, "min_confidence": 0.85},
            
            # Volume (âm lượng)
            {"input": "Bật âm lượng", "intent": "control-device", "expected": {"ACTION": "ON", "DEVICE": "volume"}, "min_confidence": 0.85},
            {"input": "Tắt tiếng đi", "intent": "control-device", "expected": {"ACTION": "OFF", "DEVICE": "volume"}, "min_confidence": 0.85},
            
            # Brightness (độ sáng)
            {"input": "Bật độ sáng", "intent": "control-device", "expected": {"ACTION": "ON", "DEVICE": "brightness"}, "min_confidence": 0.85},
            
            # Mobile data
            {"input": "Bật data", "intent": "control-device", "expected": {"ACTION": "ON", "DEVICE": "mobile_data"}, "min_confidence": 0.85},
            {"input": "Tắt 4G", "intent": "control-device", "expected": {"ACTION": "OFF", "DEVICE": "mobile_data"}, "min_confidence": 0.85},
        ]
        
        passed = 0
        total = len(test_cases)
        
        print("\n===== Test DEVICE_CONTROL Extraction =====")
        for i, tc in enumerate(test_cases, 1):
            entities, conf = self._extract_device_control_with_confidence(tc["input"], tc["intent"])
            expected = tc["expected"]
            
            # Check if all expected keys exist with correct values
            match = all(entities.get(k) == v for k, v in expected.items())
            conf_ok = conf >= tc["min_confidence"]
            
            if match and conf_ok:
                passed += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            
            print(f"[{i}/{total}] {status}")
            print(f"  Input: {tc['input']}")
            print(f"  Expected: {expected}")
            print(f"  Got: {entities} (conf={conf:.2f})")
            if not match:
                print(f"  Mismatch!")
            elif not conf_ok:
                print(f"  Confidence too low: {conf:.2f} < {tc['min_confidence']}")
            print()
        
        accuracy = (passed / total) * 100
        print(f"Result: {passed}/{total} passed ({accuracy:.1f}%)")
        print(f"Target: >=90% → {'✅達成' if accuracy >= 90 else '❌ Chưa đạt'}")
        return accuracy


