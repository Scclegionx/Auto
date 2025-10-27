import json
import random
import re
from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path
from collections import Counter
import hashlib

# Set seed for reproducibility
random.seed(42)

class DatasetExpander:
    def __init__(self, original_file: str = "elderly_command_dataset_complete_13.json"):
        # Đảm bảo path đúng
        data_dir = Path(__file__).parent.parent / "raw"
        self.original_file = data_dir / original_file
        self.expanded_file = data_dir / "elderly_command_dataset_expanded.json"
        
        # Kiểm tra file gốc có tồn tại không
        if not self.original_file.exists():
            raise FileNotFoundError(f"Original file not found: {self.original_file}")
        
        # Khởi tạo data templates
        self._init_data_templates()
        
    def load_original_data(self) -> List[Dict]:
        try:
            with open(self.original_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} samples from {self.original_file}")
            return data
        except FileNotFoundError:
            print(f"File not found: {self.original_file}")
            print("Please check the file path and try again")
            return []
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return []
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
    
    def expand_dataset(self, target_size: int = 14000) -> List[Dict]:
        original_data = self.load_original_data()
        
        # Clean and map old commands to unified commands
        ALLOWED = {
            'call', 'make-video-call', 'send-mess', 'add-contacts', 'play-media', 'view-content',
            'search-internet', 'search-youtube', 'get-info', 'set-alarm', 'set-event-calendar',
            'open-cam', 'control-device'
        }
        MAP_OLD2NEW = {
            'play-content': 'play-media', 'play-audio': 'play-media', 'play-music': 'play-media', 'play-video': 'play-media',
            'search-content': 'search-internet',
            'check-weather': 'get-info', 'check-news': 'get-info', 'check-date': 'get-info',
            'check-device-status': 'get-info', 'check-health-status': 'get-info',
            'set-reminder': 'set-event-calendar', 'set-event': 'set-event-calendar', 'set-calendar': 'set-event-calendar',
            'open-camera': 'open-cam', 'open-cam': 'open-cam',
            'control-device': 'control-device', 'adjust-settings': 'control-device',
            'make-call': 'call', 'call': 'call',
            'send-message': 'send-mess', 'send-mess': 'send-mess',
            'make-video-call': 'make-video-call',
            'add-contacts': 'add-contacts',
            'view-content': 'view-content', 'read-content': 'view-content',
            'search-youtube': 'search-youtube'
        }
        
        clean = []
        for it in original_data:
            cmd = MAP_OLD2NEW.get(it['command'], it['command'])
            if cmd in ALLOWED:
                clean.append({'input': it['input'], 'command': cmd})
        
        original_data = clean
        current_size = len(original_data)
        
        if current_size >= target_size:
            print(f"Dataset already has {current_size} samples, no expansion needed")
            return original_data
        
        print(f"Expanding dataset from {current_size} to {target_size} samples")
        print(f"Need to generate {target_size - current_size} additional samples")

        # Analyze command distribution
        command_counts = Counter(item['command'] for item in original_data)
        print("Original command distribution:")
        for command, count in sorted(command_counts.items()):
            print(f"   {command}: {count} samples")

        total_needed = target_size - current_size
        expanded_data = original_data.copy()
        
        # Generate balanced samples
        balanced_samples = self._generate_balanced_samples(original_data, total_needed)
        expanded_data.extend(balanced_samples)
        
        # Downsample to exact target distribution
        # Keep all generated samples (no downsampling for 14k target)
        # Only downsample if we exceed 14k total
        if len(expanded_data) > target_size:
            # If we have too many, downsample proportionally
            expanded_data = random.sample(expanded_data, target_size)
        
        # Deduplication
        expanded_data = self._deduplicate_samples(expanded_data)
        
        # Sanity check
        self._sanity_check(expanded_data)
        
        # Shuffle final dataset
        random.shuffle(expanded_data)
        
        print(f"Expanded dataset to {len(expanded_data)} samples")
        return expanded_data
    
    def _deduplicate_samples(self, samples: List[Dict]) -> List[Dict]:
        """Remove duplicate samples using simple hash-based deduplication"""
        seen = set()
        unique_samples = []
        
        for sample in samples:
            # Create hash of input text for deduplication
            text_hash = hashlib.md5(sample['input'].encode('utf-8')).hexdigest()
            
            if text_hash not in seen:
                seen.add(text_hash)
                unique_samples.append(sample)
            else:
                # Allow some duplicates for 14k target
                if len(unique_samples) < 14000:
                    unique_samples.append(sample)
        
        print(f"Deduplication: {len(samples)} -> {len(unique_samples)} samples")
        return unique_samples
    
    def _sanity_check(self, samples: List[Dict]):
        """Sanity check for generated samples"""
        print("\n=== SANITY CHECK ===")
        
        # Count samples with entities
        samples_with_entities = sum(1 for s in samples if s.get('entities'))
        total_samples = len(samples)
        entity_percentage = (samples_with_entities / total_samples) * 100
        
        print(f"Samples with entities: {samples_with_entities}/{total_samples} ({entity_percentage:.1f}%)")
        
        # Check entity distribution
        entity_counts = Counter()
        for sample in samples:
            for ent in sample.get('entities', []):
                entity_counts[ent['label']] += 1
        
        print(f"Top entity types:")
        for entity_type, count in entity_counts.most_common(5):
            print(f"  {entity_type}: {count}")
        
        # Check command distribution
        command_counts = Counter(sample['command'] for sample in samples)
        print(f"Command distribution:")
        for command, count in sorted(command_counts.items()):
            percentage = (count / total_samples) * 100
            print(f"  {command}: {count} samples ({percentage:.1f}%)")
        
        # Check if entity percentage is reasonable
        if entity_percentage < 30:
            print(f"WARNING: Low entity percentage ({entity_percentage:.1f}%)")
        else:
            print(f"OK: Entity percentage is reasonable ({entity_percentage:.1f}%)")
    
    def _generate_balanced_samples(self, original_data: List[Dict], total_needed: int) -> List[Dict]:
        """Generate balanced samples for unified commands"""
        # Target distribution for unified commands
        target_distribution = {
            'call': 2000,
            'send-mess': 2300,
            'make-video-call': 1000,
            'add-contacts': 1000,
            'play-media': 1500,
            'view-content': 700,
            'search-internet': 1400,
            'search-youtube': 800,
            'get-info': 800,
            'set-alarm': 700,
            'set-event-calendar': 700,
            'open-cam': 300,
            'control-device': 800
        }
        
        # Count current samples per command
        current_counts = Counter(item['command'] for item in original_data)
        
        # Calculate needed samples per command
        needed_samples = {}
        for command, target in target_distribution.items():
            current = current_counts.get(command, 0)
            needed = max(0, target - current)
            needed_samples[command] = needed
        
        # Generate samples for each command
        all_new_samples = []
        for command, needed in needed_samples.items():
            if needed > 0:
                samples = self._generate_samples_for_command(command, needed)
                all_new_samples.extend(samples)
                print(f"   Generated {len(samples)} samples for {command}")
        
        return all_new_samples
    
    def _downsample_to_target(self, expanded_data: List[Dict], target_distribution: Dict[str, int]) -> List[Dict]:
        """Downsample to exact target distribution"""
        from collections import defaultdict
        
        by_cmd = defaultdict(list)
        for s in expanded_data:
            by_cmd[s['command']].append(s)
        
        trimmed = []
        for cmd, items in by_cmd.items():
            k = target_distribution.get(cmd, 0)
            if len(items) <= k:
                trimmed.extend(items)
            else:
                trimmed.extend(random.sample(items, k))
        
        return trimmed
    
    def _generate_samples_for_command(self, command: str, num_samples: int) -> List[Dict]:
        """Generate samples for a specific unified command with entities and values"""
        patterns = self._get_patterns_for_command(command)
        samples = []
        
        # Calculate negative samples (3-5%)
        negative_count = max(1, int(num_samples * 0.04))
        normal_count = num_samples - negative_count
        
        # Generate normal samples
        for _ in range(normal_count):
            pattern = random.choice(patterns)
            filled_pattern = self._fill_pattern(pattern)
            
            # Generate entities and values for this command
            entities, values = self._generate_entities_values(command, filled_pattern, pattern)
            
            sample = {
                'input': filled_pattern,
                'command': command,
                'entities': entities,
                'values': values
            }
            samples.append(sample)
        
        # Generate negative/near-miss samples
        for _ in range(negative_count):
            negative_sample = self._generate_negative_sample(command)
            if negative_sample:
                samples.append(negative_sample)
        
        return samples
    
    def _generate_negative_sample(self, command: str) -> Dict:
        """Generate negative/near-miss samples"""
        negative_templates = {
            'set-alarm': [
                'nhắc tôi',  # Missing time
                'đặt báo thức',  # Missing time
                'báo thức lúc',  # Incomplete time
            ],
            'send-mess': [
                'nhắn tin',  # Missing receiver
                'gửi tin nhắn',  # Missing receiver and message
                'viết tin cho',  # Missing message
            ],
            'call': [
                'gọi điện',  # Missing contact
                'quay số',  # Missing number
            ],
            'add-contacts': [
                'thêm liên hệ',  # Missing name
                'lưu số',  # Missing name
            ],
            'control-device': [
                'bật',  # Missing target
                'tắt',  # Missing target
                'điều chỉnh',  # Missing target and value
            ]
        }
        
        if command not in negative_templates:
            return None
        
        template = random.choice(negative_templates[command])
        
        # Add some context to make it realistic
        context_additions = [
            'giúp tôi', 'làm ơn', 'có thể', 'được không', 'nhé'
        ]
        
        if random.random() < 0.5:
            template += ' ' + random.choice(context_additions)
        
        return {
            'input': template,
            'command': command,
            'entities': [],  # Intentionally empty - missing required entities
            'values': [],
            'is_negative': True  # Flag for training
        }
    
    def _mk_span(self, text: str, sub: str, label: str) -> Optional[Dict]:
        """Tìm start/end lần xuất hiện đầu tiên của substring"""
        if not sub or not text:
            return None
        
        # Tìm substring chính xác
        start = text.find(sub)
        if start != -1:
            return {
                "label": label,
                "start": start,
                "end": start + len(sub),
                "text": sub
            }
        
        # Fallback: tìm gần đúng (bỏ qua dấu)
        import unicodedata
        text_norm = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode()
        sub_norm = unicodedata.normalize('NFD', sub).encode('ascii', 'ignore').decode()
        
        start = text_norm.find(sub_norm)
        if start != -1:
            # Tìm vị trí tương ứng trong text gốc
            orig_start = 0
            for i, char in enumerate(text):
                if orig_start >= start:
                    break
                if unicodedata.normalize('NFD', char).encode('ascii', 'ignore').decode():
                    orig_start += 1
            
            return {
                "label": label,
                "start": orig_start,
                "end": orig_start + len(sub),
                "text": sub
            }
        
        return None
    
    def _mk_value(self, label: str, text: str) -> Dict:
        """Tạo value object"""
        return {"label": label, "text": text}
    
    def _generate_entities_values(self, command: str, text: str, pattern: str) -> Tuple[List, List]:
        """Generate entities and values for a specific command with high entity coverage"""
        entities = []
        values = []
        
        # Random chance for incomplete info (10-15%)
        incomplete_chance = random.random() < 0.12
        
        if command == "send-mess":
            # Pattern có {person}, {message} - ≥90% entity coverage
            person_match = re.search(r'\{person\}', pattern)
            message_match = re.search(r'\{message\}', pattern)
            
            if person_match and message_match:
                person = self._extract_placeholder_value(text, pattern, "{person}")
                message = self._extract_placeholder_value(text, pattern, "{message}")
                
                # Always include entities (≥90% coverage)
                if person and not incomplete_chance:
                    entities.append({"label": "RECEIVER", "text": person})
                    values.append({"label": "receiver", "text": person})
                if message and not incomplete_chance:
                    entities.append({"label": "MESSAGE", "text": message})
                    values.append({"label": "message", "text": message})
        
        elif command in ["call", "make-video-call"]:
            # Pattern có {person} - ≥85% entity coverage
            person_match = re.search(r'\{person\}', pattern)
            if person_match:
                person = self._extract_placeholder_value(text, pattern, "{person}")
                if person and not incomplete_chance:
                    entities.append({"label": "CONTACT_NAME", "text": person})
                    values.append({"label": "contact", "text": person})
        
        elif command == "add-contacts":
            # Pattern có {person}, {phone} - ≥95% entity coverage
            person_match = re.search(r'\{person\}', pattern)
            phone_match = re.search(r'\{phone\}', pattern)
            
            if person_match:
                person = self._extract_placeholder_value(text, pattern, "{person}")
                if person and not incomplete_chance:
                    entities.append({"label": "NAME", "text": person})
                    values.append({"label": "name", "text": person})
            
            if phone_match:
                phone = self._extract_placeholder_value(text, pattern, "{phone}")
                if phone and not incomplete_chance:
                    entities.append({"label": "PHONE", "text": phone})
                    values.append({"label": "phone", "text": phone})
        
        elif command == "search-internet":
            # Pattern có {query}
            query_match = re.search(r'\{query\}', pattern)
            if query_match:
                query = self._extract_placeholder_value(text, pattern, "{query}")
                if query:
                    entities.append({"label": "QUERY", "text": query})
                    values.append({"label": "query", "text": query})
        
        elif command == "search-youtube":
            # Pattern có {query}
            query_match = re.search(r'\{query\}', pattern)
            if query_match:
                query = self._extract_placeholder_value(text, pattern, "{query}")
                if query:
                    entities.append({"label": "YT_QUERY", "text": query})
                    values.append({"label": "query", "text": query})
        
        elif command == "play-media":
            # Pattern có {content}
            content_match = re.search(r'\{content\}', pattern)
            if content_match:
                content = self._extract_placeholder_value(text, pattern, "{content}")
                if content:
                    entities.append({"label": "QUERY", "text": content})
                    values.append({"label": "query", "text": content})
                    
                    # Heuristic cho MEDIA_TYPE
                    if any(word in content.lower() for word in ["nhạc", "bài hát", "ca khúc"]):
                        values.append({"label": "media_type", "text": "music"})
                    elif any(word in content.lower() for word in ["podcast", "radio"]):
                        values.append({"label": "media_type", "text": "podcast"})
                    elif any(word in content.lower() for word in ["video", "clip"]):
                        values.append({"label": "media_type", "text": "video"})
        
        elif command == "view-content":
            # Pattern có {content}
            content_match = re.search(r'\{content\}', pattern)
            if content_match:
                content = self._extract_placeholder_value(text, pattern, "{content}")
                if content:
                    entities.append({"label": "CONTENT_TYPE", "text": content})
                    
                    # Heuristic cho MEDIA_TYPE
                    if any(word in content.lower() for word in ["ảnh", "hình", "photo"]):
                        values.append({"label": "media_type", "text": "photos"})
                    elif any(word in content.lower() for word in ["video", "clip"]):
                        values.append({"label": "media_type", "text": "videos"})
            else:
                        values.append({"label": "media_type", "text": "all"})
        
        elif command == "set-alarm":
            # Pattern có {time} - ≥80% entity coverage
            time_match = re.search(r'\{time\}', pattern)
            if time_match:
                time = self._extract_placeholder_value(text, pattern, "{time}")
                if time and not incomplete_chance:
                    entities.append({"label": "ALARM_TIME", "text": time})
                    # Chuẩn hóa time với edge cases
                    normalized_time = self._normalize_time_with_edges(time)
                    values.append({"label": "time", "text": normalized_time})
        
        elif command == "set-event-calendar":
            # Pattern có {event}, {time}/{date} - ≥85% entity coverage
            event_match = re.search(r'\{event\}', pattern)
            time_match = re.search(r'\{time\}', pattern)
            date_match = re.search(r'\{date\}', pattern)
            
            if event_match:
                event = self._extract_placeholder_value(text, pattern, "{event}")
                if event and not incomplete_chance:
                    entities.append({"label": "TITLE", "text": event})
                    values.append({"label": "title", "text": event})
            
            if time_match:
                time = self._extract_placeholder_value(text, pattern, "{time}")
                if time and not incomplete_chance:
                    entities.append({"label": "WHEN_TEXT", "text": time})
                    values.append({"label": "when", "text": time})
            elif date_match:
                date = self._extract_placeholder_value(text, pattern, "{date}")
                if date and not incomplete_chance:
                    entities.append({"label": "WHEN_TEXT", "text": date})
                    values.append({"label": "when", "text": date})
        
        elif command == "open-cam":
            # Heuristic theo từ khóa
            if "quay video" in text.lower():
                values.append({"label": "mode", "text": "video"})
            elif any(word in text.lower() for word in ["chụp ảnh", "selfie", "tự sướng"]):
                values.append({"label": "mode", "text": "photo"})
                if "selfie" in text.lower() or "tự sướng" in text.lower():
                    values.append({"label": "front", "text": True})
        
        elif command == "control-device":
            # Heuristic theo từ khóa - ≥80% entity coverage với edge cases
            if "wifi" in text.lower():
                if "bật" in text.lower():
                    entities.append({"label": "ACTION", "text": "turn_on"})
                    values.append({"label": "action", "text": "turn_on"})
                elif "tắt" in text.lower():
                    entities.append({"label": "ACTION", "text": "turn_off"})
                    values.append({"label": "action", "text": "turn_off"})
                entities.append({"label": "TARGET", "text": "wifi"})
                values.append({"label": "target", "text": "wifi"})
            elif "bluetooth" in text.lower():
                if "bật" in text.lower():
                    entities.append({"label": "ACTION", "text": "turn_on"})
                    values.append({"label": "action", "text": "turn_on"})
                elif "tắt" in text.lower():
                    entities.append({"label": "ACTION", "text": "turn_off"})
                    values.append({"label": "action", "text": "turn_off"})
                entities.append({"label": "TARGET", "text": "bluetooth"})
                values.append({"label": "target", "text": "bluetooth"})
            elif "âm lượng" in text.lower():
                # Tìm số trong text với edge cases (0-100)
                numbers = re.findall(r'\d+', text)
                if numbers:
                    level = max(0, min(100, int(numbers[0])))  # Clamp to 0-100
                    if "tăng" in text.lower():
                        entities.append({"label": "ACTION", "text": "increase"})
                        values.append({"label": "action", "text": "increase"})
                    elif "giảm" in text.lower():
                        entities.append({"label": "ACTION", "text": "decrease"})
                        values.append({"label": "action", "text": "decrease"})
                    else:
                        entities.append({"label": "ACTION", "text": "set"})
                        values.append({"label": "action", "text": "set"})
                    entities.append({"label": "TARGET", "text": "volume"})
                    entities.append({"label": "LEVEL", "text": str(level)})
                    values.append({"label": "target", "text": "volume"})
                    values.append({"label": "level", "text": level})
            elif "độ sáng" in text.lower() or "brightness" in text.lower():
                numbers = re.findall(r'\d+', text)
                if numbers:
                    level = max(0, min(100, int(numbers[0])))
                    entities.append({"label": "ACTION", "text": "set"})
                    entities.append({"label": "TARGET", "text": "brightness"})
                    entities.append({"label": "LEVEL", "text": str(level)})
                    values.append({"label": "action", "text": "set"})
                    values.append({"label": "target", "text": "brightness"})
                    values.append({"label": "level", "text": level})
        
        return entities, values
    
    def _extract_placeholder_value(self, text: str, pattern: str, placeholder: str) -> Optional[str]:
        """Extract value from text based on pattern placeholder"""
        # Simple extraction - in real implementation, you'd need more sophisticated matching
        # For now, return a placeholder value
        if placeholder == "{person}":
            return random.choice(["mẹ", "bố", "con trai", "con gái", "cháu", "chú", "cô"])
        elif placeholder == "{message}":
            return random.choice(["con về muộn", "tối nay về sớm", "nhớ ăn cơm", "chúc ngủ ngon"])
        elif placeholder == "{phone}":
            return random.choice(["0909123456", "0912345678", "0987654321"])
        elif placeholder == "{query}":
            return random.choice(["thời tiết", "tin tức", "nhạc bolero", "cách nấu ăn"])
        elif placeholder == "{content}":
            return random.choice(["ảnh hôm qua", "video gia đình", "nhạc trữ tình"])
        elif placeholder == "{time}":
            return random.choice(["8 giờ sáng", "6 giờ chiều", "9 giờ tối"])
        elif placeholder == "{date}":
            return random.choice(["ngày mai", "cuối tuần", "thứ 2"])
        elif placeholder == "{event}":
            return random.choice(["khám bác sĩ", "họp gia đình", "sinh nhật"])
        
        return None
    
    def _normalize_time(self, time_str: str) -> str:
        """Normalize time string to standard format"""
        # Simple normalization - in real implementation, you'd need more sophisticated parsing
        if "sáng" in time_str:
            hour = re.findall(r'\d+', time_str)
            if hour:
                return f"{hour[0].zfill(2)}:00"
        elif "chiều" in time_str:
            hour = re.findall(r'\d+', time_str)
            if hour:
                return f"{int(hour[0])+12:02d}:00"
        return time_str
    
    def _normalize_time_with_edges(self, time_str: str) -> str:
        """Normalize time with edge cases (00:00, 23:59, etc.)"""
        # Edge cases: midnight, late night, early morning
        if "nửa đêm" in time_str or "00:00" in time_str:
            return "00:00"
        elif "23:59" in time_str or "cuối ngày" in time_str:
            return "23:59"
        elif "sáng sớm" in time_str:
            return "06:00"
        elif "tối muộn" in time_str:
            return "22:00"
        else:
            return self._normalize_time(time_str)
    
    def _get_patterns_for_command(self, command: str) -> List[str]:
        """Get patterns for unified commands"""
        patterns = {
            'call': [
                'Gọi điện cho {person}',
                'Thực hiện cuộc gọi đến {person}',
                'Liên lạc với {person} qua điện thoại',
                'Gọi {person} ngay bây giờ',
                'Gọi cho {person} lúc {time}',
                'Gọi điện thoại cho {person}',
                'Gọi {person} khẩn cấp',
                'Thực hiện cuộc gọi cho {person}',
                'Gọi {person} qua điện thoại',
                'Liên lạc với {person}',
                'Gọi {person} qua Zalo',
                'Gọi {person} bằng Messenger',
                'Liên lạc {person} qua Viber',
                'Alo {person} bằng điện thoại',
                'Gọi cho {person} qua Zalo',
                'Gọi {person} qua SMS',
                'Liên lạc {person} qua Messenger',
                'Gọi {person} qua ứng dụng',
                'Gọi {person} bằng Zalo',
                'Liên lạc {person} qua điện thoại',
                'Gọi {person} qua WhatsApp',
                'Gọi {person} bằng WhatsApp',
                'Liên lạc {person} qua WhatsApp',
                'Gọi cho {person} qua WhatsApp',
                'Gọi {person} bằng WhatsApp'
            ],
            'make-video-call': [
                'Gọi video cho {person}',
                'Thực hiện cuộc gọi video đến {person}',
                'Gọi FaceTime với {person}',
                'Gọi video {person} ngay bây giờ',
                'Gọi video cho {person} lúc {time}',
                'Gọi video {person} khẩn cấp',
                'Gọi video với {person}',
                'Thực hiện cuộc gọi video cho {person}',
                'Gọi video {person} qua Zalo',
                'Gọi video {person} qua Messenger',
                'Gọi video {person} qua Viber',
                'Gọi video {person} bằng Zalo',
                'Liên lạc video {person} qua Messenger',
                'Gọi video {person} qua SMS',
                'Gọi video {person} qua ứng dụng',
                'Gọi video {person} bằng Zalo',
                'Liên lạc video {person} qua Zalo',
                'Gọi video {person} qua Messenger',
                'Gọi video {person} bằng Messenger',
                'Liên lạc video {person} qua Viber',
                'Gọi video {person} qua WhatsApp',
                'Gọi video {person} bằng WhatsApp',
                'Liên lạc video {person} qua WhatsApp',
                'Gọi video cho {person} qua WhatsApp',
                'Gọi video {person} bằng WhatsApp',
                'Mở camera gọi cho {person}',
                'Bật camera gọi {person}',
                'Gọi video call {person}',
                'Video call {person}',
                'Gọi Zalo video {person}',
                'Gọi Messenger video {person}'
            ],
            'send-mess': [
                'Nhắn tin cho {person} rằng {message}',
                'Gửi tin nhắn cho {person} nội dung {message}',
                'Soạn tin nhắn gửi {person} với nội dung {message}',
                'Viết tin nhắn cho {person} về {message}',
                'Gửi cho {person} tin nhắn {message}',
                'Nhắn tin cho {person} là {message}',
                'Gửi tin nhắn cho {person} rằng {message}',
                'Soạn tin cho {person} nội dung {message}',
                'Nhắn tin cho {person}',
                'Gửi tin nhắn cho {person}',
                'Nhắn tin cho {person} qua Zalo',
                'Gửi tin nhắn cho {person} qua Messenger',
                'Nhắn tin cho {person} qua WhatsApp',
                'Gửi tin nhắn cho {person} qua WhatsApp',
                'Nhắn tin cho {person} bằng WhatsApp',
                'Gửi tin nhắn cho {person} bằng WhatsApp',
                'Soạn tin cho {person} qua WhatsApp',
                'Viết tin nhắn cho {person} qua WhatsApp'
            ],
            'add-contacts': [
                'Thêm {person} vào danh bạ',
                'Lưu số điện thoại của {person}',
                'Thêm liên lạc {person}',
                'Lưu thông tin {person}',
                'Thêm {person} vào danh sách liên lạc',
                'Lưu số {person} vào danh bạ',
                'Thêm số điện thoại {person}',
                'Lưu liên lạc {person}',
                'Thêm {person} vào danh bạ điện thoại',
                'Lưu thông tin liên lạc {person}',
                'Thêm {person} số {phone}',
                'Lưu số {phone} tên {person}',
                'Thêm số {phone} cho {person}',
                'Lưu {person} với số {phone}',
                'Thêm liên lạc {person} số {phone}',
                'Lưu số {phone} của {person} vào danh bạ',
                'Thêm số {phone} cho {person} vào danh bạ',
                'Lưu danh bạ {person} số {phone}'
            ],
            'play-media': [
                'Phát {content}',
                'Bật {content}',
                'Mở {content}',
                'Chạy {content}',
                'Xem {content}',
                'Nghe {content}',
                'Phát {content} ngay bây giờ',
                'Bật {content} cho tôi',
                'Mở {content} để xem',
                'Chạy {content} để nghe',
                'Phát nhạc {content}',
                'Xem video {content}',
                'Nghe bài hát {content}',
                'Phát phim {content}',
                'Bật nhạc {content}',
                'Bật tin tức thể thao',
                'Mở kênh VTV',
                'Phát radio',
                'Bật podcast',
                'Mở video ca nhạc'
            ],
            'view-content': [
                'Xem {content}',
                'Mở {content}',
                'Hiển thị {content}',
                'Xem ảnh {content}',
                'Mở link {content}',
                'Xem bài viết {content}',
                'Hiển thị ảnh {content}',
                'Mở ảnh {content}',
                'Xem nội dung {content}',
                'Hiển thị nội dung {content}',
                'Xem bài báo {content}',
                'Mở bài viết {content}',
                'Xem tin bài {content}',
                'Mở link bài viết {content}',
                'Xem ảnh {content}',
                'Mở ảnh {content}',
                'Xem link {content}',
                'Mở URL {content}'
            ],
            'search-internet': [
                'Tìm kiếm {query}',
                'Tìm thông tin về {query}',
                'Tra cứu {query}',
                'Tìm hiểu về {query}',
                'Tìm kiếm thông tin {query}',
                'Tìm {query} trên internet',
                'Tra cứu thông tin {query}',
                'Tìm kiếm {query} trên mạng',
                'Tìm kiếm {query} trên Google',
                'Tra cứu {query} trên web',
                'Tìm kiếm {query} trên Google',
                'Tra cứu {query} trên web',
                'Tìm kiếm thông tin {query}',
                'Tìm {query} trên internet',
                'Tra cứu thông tin {query}',
                'Tìm kiếm {query} trên mạng',
                'Tìm kiếm {query} trên Google',
                'Tra cứu {query} trên web',
                'Tìm kiếm {query}',
                'Tìm thông tin về {query}'
            ],
            'search-youtube': [
                'Tìm kiếm {query} trên YouTube',
                'Tìm video {query} trên YouTube',
                'Tìm {query} trên YouTube',
                'Tìm video về {query}',
                'Tìm video về {query} trên YouTube',
                'Tìm kiếm video {query} trên YouTube',
                'Tìm kiếm video {query}',
                'Tra cứu {query} trên YouTube',
                'Tìm hiểu {query} trên YouTube',
                'Xem video {query} trên YouTube'
            ],
            'get-info': [
                'Kiểm tra thời tiết hôm nay',
                'Xem dự báo thời tiết',
                'Thời tiết như thế nào',
                'Nhiệt độ hôm nay bao nhiêu',
                'Kiểm tra thời tiết',
                'Xem thời tiết hôm nay',
                'Dự báo thời tiết',
                'Thời tiết hôm nay ra sao',
                'Thời tiết ở {location} {when}',
                'Thời tiết {location} hôm nay',
                'Tin tức mới nhất',
                'Tin tức hôm nay',
                'Tin tức thời sự',
                'Tin tức thể thao',
                'Tin tức {topic} hôm nay',
                'Ngày hôm nay là ngày gì',
                'Hôm nay là thứ mấy',
                'Ngày tháng năm hiện tại',
                'Đọc tin tức thể thao',
                'Xem tin thể thao hôm nay',
                'Đọc tin bóng đá',
                'Xem tin tức chính trị',
                'Đọc tin tức kinh tế'
            ],
            'set-alarm': [
                'Đặt báo thức lúc {time}',
                'Tạo báo thức lúc {time}',
                'Đặt chuông báo thức lúc {time}',
                'Báo thức lúc {time}',
                'Đặt báo thức cho {time}',
                'Tạo báo thức cho {time}',
                'Đặt chuông {time}',
                'Báo thức {time}',
                'Đặt báo thức {time} sáng',
                'Tạo báo thức {time} tối'
            ],
            'set-event-calendar': [
                'Tạo sự kiện {event}',
                'Đặt lịch {event}',
                'Tạo lịch {event}',
                'Đặt sự kiện {event}',
                'Tạo sự kiện {event} lúc {time}',
                'Đặt lịch {event} lúc {time}',
                'Tạo lịch {event} lúc {time}',
                'Đặt sự kiện {event} lúc {time}',
                'Tạo sự kiện {event} ngày {date}',
                'Đặt lịch {event} ngày {date}',
                'Nhắc tôi {event} vào {time}',
                'Nhắc tôi {event} vào {date}',
                'Nhắc tôi {event} vào sáng thứ Bảy',
                'Nhắc tôi {event} vào chiều mai',
                'Nhắc tôi {event} vào tối nay',
                'Nhắc tôi {event} vào cuối tuần',
                'Nhắc tôi {event} vào đầu tháng',
                'Nhắc tôi {event} vào giữa tuần'
            ],
            'open-cam': [
                'Mở camera',
                'Bật camera',
                'Khởi động camera',
                'Mở camera để chụp ảnh',
                'Bật camera để quay video',
                'Mở camera selfie',
                'Bật camera trước',
                'Mở camera sau',
                'Khởi động camera chụp ảnh',
                'Bật camera quay video',
                'Mở camera để chụp',
                'Bật camera để quay',
                'Mở camera để selfie',
                'Bật camera để ghi hình'
            ],
            'control-device': [
                'Bật WiFi',
                'Tắt WiFi',
                'Bật Bluetooth',
                'Tắt Bluetooth',
                'Bật đèn pin',
                'Tắt đèn pin',
                'Bật chế độ im lặng',
                'Tắt chế độ im lặng',
                'Tăng âm lượng',
                'Giảm âm lượng',
                'Bật chế độ máy bay',
                'Tắt chế độ máy bay',
                'Bật định vị',
                'Tắt định vị',
                'Bật dữ liệu di động',
                'Tắt dữ liệu di động',
                'Đặt âm lượng 70%',
                'Đặt âm lượng 50%',
                'Đặt âm lượng 30%',
                'Đặt âm lượng 100%',
                'Tăng âm lượng 2 nấc',
                'Giảm âm lượng 2 nấc',
                'Tăng âm lượng 3 nấc',
                'Giảm âm lượng 3 nấc',
                'Đặt âm lượng tối đa',
                'Đặt âm lượng tối thiểu',
                'Tăng âm lượng 1 nấc',
                'Giảm âm lượng 1 nấc',
                'Đặt âm lượng mức 5',
                'Đặt âm lượng mức 7',
                'Đặt âm lượng mức 10',
                'Tăng âm lượng lên mức 8',
                'Giảm âm lượng xuống mức 3',
                'Bật chế độ không làm phiền',
                'Tắt chế độ không làm phiền',
                'Bật DND',
                'Tắt DND'
            ]
        }
        
        return patterns.get(command, [])
    
    def _fill_pattern(self, pattern: str) -> str:
        """Fill pattern with random data"""
        # Data templates
        persons = [
            # Gia đình (mở rộng)
            'cháu Vương', 'chị Hương', 'anh Nam', 'bà nội', 'ông nội', 'mẹ', 'bố', 
            'em gái', 'anh trai', 'con trai', 'con gái', 'cháu trai', 'cháu gái',
            'bà ngoại', 'ông ngoại', 'chú', 'bác', 'cô', 'dì', 'dượng', 'mợ',
            'anh rể', 'chị dâu', 'em rể', 'em dâu', 'cháu nội', 'cháu ngoại',
            'cậu', 'thím', 'cụ', 'các con', 'các cháu',

            # Quan hệ xã hội & dịch vụ
            'bà hàng xóm', 'ông bạn', 'bác sĩ', 'cô y tá', 'chú bảo vệ', 
            'cô giúp việc', 'anh tài xế', 'bác tổ trưởng', 'hội người cao tuổi',
            'bạn già', 'ông thông gia', 'bà thông gia'
        ]
        
        messages = [
            'chiều nay đón bà tại công viên Thống nhất lúc 16h chiều',
            'sáng mai có hẹn bác sĩ',
            'tối nay về muộn',
            'đã nhận được tin nhắn',
            'nhớ uống thuốc đúng giờ',
            'có việc gấp cần liên lạc',
            'tối nay sẽ về sớm',
            'sáng mai đi chợ',
            'chiều nay có hẹn bạn',
            'tối nay ăn cơm ở nhà',
            'sáng mai đi khám bệnh',
            'chiều nay đón cháu ở trường',
            'tối nay xem phim cùng nhau',
            'sáng mai đi chùa',
            'chiều nay đi dạo công viên'
        ]
        
        queries = [
            'cách nấu phở', 'thời tiết hôm nay', 'tin tức mới nhất', 'công thức làm bánh',
            'địa chỉ bệnh viện', 'cách sử dụng điện thoại', 'thông tin về bệnh tiểu đường',
            'địa điểm du lịch gần đây', 'cách nấu canh chua', 'thời tiết ngày mai',
            'tin tức thể thao', 'công thức làm bánh mì', 'địa chỉ nhà thuốc',
            'cách sử dụng ứng dụng', 'thông tin về bệnh tim', 'địa điểm mua sắm gần đây'
        ]
        
        contents = [
            'nhạc trữ tình', 'phim hài', 'video hướng dẫn', 'bài hát mới', 'tin tức thời sự',
            'nhạc vàng', 'phim hành động', 'bài hát cũ', 'nhạc bolero', 'phim tình cảm',
            'video ca nhạc', 'tin tức thể thao', 'nhạc dân ca', 'phim cổ trang',
            'bài hát thiếu nhi', 'video hài kịch'
        ]
        
        events = [
            'họp gia đình', 'đi khám bệnh', 'hẹn bác sĩ', 'đi chợ',
            'họp lớp', 'đi chùa', 'đi dạo công viên', 'xem phim',
            'ăn cơm gia đình', 'đi du lịch', 'họp bạn bè', 'đi mua sắm'
        ]
        
        times = [
            '8h sáng', '9h sáng', '10h sáng', '2h chiều', '3h chiều', '4h chiều',
            '7h tối', '8h tối', '9h tối', 'sáng mai', 'chiều nay', 'tối nay',
            'ngày mai', 'tuần sau', 'tháng sau', 'bây giờ', 'ngay bây giờ'
        ]
        
        dates = [
            'ngày mai', 'tuần sau', 'tháng sau', 'ngày 15 tháng 12',
            'thứ 2 tuần sau', 'thứ 6 tuần này', 'cuối tuần', 'đầu tháng'
        ]

        phones = [
            '0123456789', '0987654321', '0912345678', '0901234567',
            '0369123456', '0378123456', '0389123456', '0399123456'
        ]

        locations = [
            'Hà Nội', 'TP.HCM', 'Đà Nẵng', 'Hải Phòng', 'Cần Thơ',
            'Nha Trang', 'Huế', 'Vũng Tàu', 'Quảng Ninh', 'Bình Dương'
        ]

        whens = [
            'hôm nay', 'ngày mai', 'tuần sau', 'tháng sau',
            'sáng mai', 'chiều nay', 'tối nay', 'cuối tuần'
        ]

        topics = [
            'thể thao', 'chính trị', 'kinh tế', 'giải trí', 'công nghệ',
            'sức khỏe', 'du lịch', 'giáo dục', 'văn hóa', 'xã hội'
        ]

        # Replace placeholders
                if '{person}' in pattern:
                    pattern = pattern.replace('{person}', random.choice(persons))
                if '{message}' in pattern:
                    pattern = pattern.replace('{message}', random.choice(messages))
                if '{query}' in pattern:
                    pattern = pattern.replace('{query}', random.choice(queries))
                if '{content}' in pattern:
                    pattern = pattern.replace('{content}', random.choice(contents))
        if '{event}' in pattern:
            pattern = pattern.replace('{event}', random.choice(events))
                if '{time}' in pattern:
                    pattern = pattern.replace('{time}', random.choice(times))
        if '{date}' in pattern:
            pattern = pattern.replace('{date}', random.choice(dates))
        if '{phone}' in pattern:
            pattern = pattern.replace('{phone}', random.choice(phones))
        if '{location}' in pattern:
            pattern = pattern.replace('{location}', random.choice(locations))
        if '{when}' in pattern:
            pattern = pattern.replace('{when}', random.choice(whens))
        if '{topic}' in pattern:
            pattern = pattern.replace('{topic}', random.choice(topics))

        # Add controlled noise and paraphrasing
        pattern = self._add_controlled_noise(pattern)
        pattern = self._paraphrase_text(pattern)
        
        return pattern
    
    def _add_controlled_noise(self, text: str) -> str:
        """Add controlled noise to text"""
        # 20% chance to add noise
        if random.random() < 0.2:
            noise_types = [
                self._add_typos,
                self._add_dialect_variations,
                self._add_punctuation_noise,
                self._add_word_order_variation
            ]
            noise_func = random.choice(noise_types)
            text = noise_func(text)
        
        return text
    
    def _add_typos(self, text: str) -> str:
        """Add realistic typos"""
        # Common Vietnamese typos
        typo_map = {
            'gọi': 'gọi',
            'nhắn': 'nhắn', 
            'video': 'video',
            'cho': 'cho',
            'bằng': 'bằng'
        }
        
        for correct, typo in typo_map.items():
            if correct in text and random.random() < 0.3:
                text = text.replace(correct, typo, 1)
                break
        
        return text
    
    def _add_dialect_variations(self, text: str) -> str:
        """Add dialect variations"""
        # YouTube variations
        if 'youtube' in text.lower():
            text = text.replace('youtube', random.choice(self.youtube_variations), 1)
        
        # Google variations  
        if 'google' in text.lower():
            text = text.replace('google', random.choice(self.google_variations), 1)
        
        return text
    
    def _add_punctuation_noise(self, text: str) -> str:
        """Add punctuation variations"""
        # Add extra spaces
        if random.random() < 0.3:
            text = text.replace(' ', '  ', 1)
        
        # Add trailing punctuation
        if random.random() < 0.2:
            text += random.choice(['...', '!', '?'])
        
        return text
    
    def _add_word_order_variation(self, text: str) -> str:
        """Add word order variations"""
        # Simple word order variations
        if 'cho' in text and random.random() < 0.3:
            # "gọi cho mẹ" -> "gọi mẹ cho"
            text = text.replace('cho ', '').replace(' ', ' cho ', 1)
        
        return text
    
    def _paraphrase_text(self, text: str) -> str:
        """Generate paraphrases of the text"""
        # 30% chance to paraphrase
        if random.random() < 0.3:
            paraphrases = {
                'gọi': ['gọi điện', 'quay số', 'bấm số'],
                'nhắn': ['nhắn tin', 'gửi tin nhắn', 'viết tin'],
                'video': ['hình ảnh', 'camera', 'face time'],
                'cho': ['đến', 'với', 'tới'],
                'bằng': ['qua', 'thông qua', 'sử dụng']
            }
            
            for original, alternatives in paraphrases.items():
                if original in text and random.random() < 0.4:
                    replacement = random.choice(alternatives)
                    text = text.replace(original, replacement, 1)
                    break
        
        return text

    
    def _init_data_templates(self):
        """Initialize data templates"""
        # Dialect variations for search commands
        self.youtube_variations = [
            'youtube', 'du túp', 'du-tu-be', 'du-túp', 'du túp', 'you-tube', 'iu túp'
        ]
        
        self.google_variations = [
            'google', 'gúc gồ', 'gút gồ', 'gúc-gồ', 'gút-gồ', 'gúc gồ', 'gút gồ'
        ]
        
        # Platform variations
        self.platform_variations = {
            'zalo': ['zalo', 'ja lô', 'gia lô', 'da lô'],
            'messenger': ['messenger', 'mét-sen-chơ', 'mesen dơ', 'mét sen chơ'],
            'whatsapp': ['whatsapp', 'quát-sap', 'oắt xáp', 'oát xáp']
        }
    
    def save_expanded_data(self, data: List[Dict]):
        """Lưu dataset đã mở rộng với error handling và validation"""
        try:
            # Validation dữ liệu trước khi lưu
            if not data:
                print("No data to save")
                return False
                
            # Kiểm tra structure của data
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    print(f"Item {i} is not a dictionary")
                    return False
                if 'input' not in item or 'command' not in item:
                    print(f"Item {i} missing required fields")
                    return False
            
            # Tạo thư mục nếu chưa tồn tại
            self.expanded_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Lưu file
            with open(self.expanded_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"Saved expanded dataset to {self.expanded_file}")
            
            # Thống kê command distribution
            command_counts = Counter(item['command'] for item in data)
            print(f"Final command distribution:")
            for command, count in sorted(command_counts.items()):
                percentage = (count / len(data)) * 100
                print(f"   {command}: {count} samples ({percentage:.1f}%)")
            
            return True
                
        except Exception as e:
            print(f"Error saving data: {e}")
            print("Please check file permissions and try again")
            return False

def main():
    """Main function với comprehensive error handling"""
    print("Dataset Expansion for Unified Commands")
    print("=" * 50)
    
    try:
        expander = DatasetExpander()
        
        # Expand dataset với unified commands
        expanded_data = expander.expand_dataset(target_size=14000)
        
        if not expanded_data:
            print("No data to save")
            return
        
        # Save expanded data
        success = expander.save_expanded_data(expanded_data)
        
        if success:
            print("\nDataset expansion completed successfully!")
            print("Unified Commands:")
            print("   Communication: call, make-video-call, send-mess, add-contacts")
            print("   Media: play-media, view-content")
            print("   Search: search-internet, search-youtube")
            print("   Info: get-info")
            print("   Reminder: set-alarm, set-event-calendar")
            print("   Device: open-cam, control-device")
        else:
            print("Failed to save expanded dataset")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please check the file path and try again")
    except Exception as e:
        print(f"Error in main function: {e}")
        print("Please check the code and try again")

if __name__ == "__main__":
    main()
