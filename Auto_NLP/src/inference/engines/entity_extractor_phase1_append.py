
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
        
        # Step 1: Extract PLATFORM (qua|bằng|trên <keyword>) và loại khỏi chuỗi parse
        platform_keywords = {
            "zalo": "zalo", "messenger": "messenger", "sms": "sms", "imessage": "imessage",
            "viber": "viber", "telegram": "telegram", "whatsapp": "whatsapp", "facebook": "facebook",
            "fb": "facebook", "mess": "messenger", "za": "zalo"
        }
        platform_pattern = r"\b(qua|bằng|trên|bang|tren)\s+(\w+)"
        platform_match = re.search(platform_pattern, text_lower)
        if platform_match:
            platform_raw = platform_match.group(2)
            for keyword, canonical in platform_keywords.items():
                if keyword in platform_raw:
                    entities["PLATFORM"] = canonical
                    # Loại platform khỏi chuỗi để parse receiver/message
                    text_lower = text_lower.replace(platform_match.group(0), " ").strip()
                    break
        
        # Step 2: Tìm marker nội dung
        content_markers = [
            r"\s+rằng\s+",  # "nhắn cho Lan rằng …"
            r"\s+là\s+",     # "nhắn cho Lan là …"
            r"\s+nói\s+",    # "cho Lan nói …"
            r"\s+bảo\s+",    # "cho Lan bảo …"
            r"\s+:",         # "cho Lan: …"
        ]
        
        marker_pos = None
        marker_text = None
        for marker_re in content_markers:
            match = re.search(marker_re, text_lower)
            if match:
                marker_pos = match.start()
                marker_text = match.group(0).strip()
                break
        
        # Step 3: Tìm cụm receiver (sau "cho/tới/đến" hoặc ngay sau verb nhắn/gửi)
        receiver_trigger_pattern = r"\b(cho|tới|đến|den|toi)\s+"
        receiver_match = re.search(receiver_trigger_pattern, text_lower)
        
        if receiver_match:
            receiver_start = receiver_match.end()
            
            # Case A: có marker sau receiver
            if marker_pos and marker_pos > receiver_start:
                receiver_text_raw = text_lower[receiver_start:marker_pos].strip()
                message_start_pos = text_lower.find(marker_text, marker_pos) + len(marker_text)
                message_text_raw = text_lower[message_start_pos:].strip()
                
                # Clean receiver: cắt tối đa 3-4 tokens
                receiver_tokens = receiver_text_raw.split()
                receiver_clean = " ".join(receiver_tokens[:4]) if len(receiver_tokens) <= 4 else " ".join(receiver_tokens[:3])
                
                entities["RECEIVER"] = receiver_clean.strip()
                entities["MESSAGE"] = message_text_raw.strip()
                confidence = 1.0  # Case A: rõ ràng nhất
            
            # Case B: không marker, dùng heuristic split
            elif not marker_pos:
                receiver_text_raw = text_lower[receiver_start:].strip()
                receiver_tokens = receiver_text_raw.split()
                
                # Heuristic: receiver tối đa 3-4 tokens, phần sau là MESSAGE
                # Dừng receiver khi gặp động từ nội dung (về, đến, ăn, mua, đón, khỏe, nhớ, uống…)
                content_verbs = ["về", "đến", "ăn", "mua", "đón", "khỏe", "nhớ", "uống", "học", "làm", "đi", "ở", "không"]
                receiver_end_idx = 3  # mặc định 3 tokens
                for idx, token in enumerate(receiver_tokens):
                    if idx >= 4:  # tối đa 4 tokens
                        break
                    if token in content_verbs and idx > 0:
                        receiver_end_idx = idx
                        break
                
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
        
        return (entities, confidence)

    def test_extract_message_receiver(self):
        """Inline test cho _extract_message_receiver_with_confidence - Target: >=80% exact match"""
        test_cases = [
            # Case A: có receiver + marker
            {
                "input": "Nhắn cho chị Mai là hôm nay mẹ về muộn",
                "intent": "send-mess",
                "expected": {"RECEIVER": "chị mai", "MESSAGE": "hôm nay mẹ về muộn"},
                "min_confidence": 1.0
            },
            {
                "input": "Gửi tới anh Trường rằng con đến đón",
                "intent": "send-mess",
                "expected": {"RECEIVER": "anh trường", "MESSAGE": "con đến đón"},
                "min_confidence": 1.0
            },
            {
                "input": "Nhắn cho Lan: chiều nay con đến",
                "intent": "send-mess",
                "expected": {"RECEIVER": "lan", "MESSAGE": "chiều nay con đến"},
                "min_confidence": 1.0
            },
            # Case B: có receiver nhưng không marker
            {
                "input": "Nhắn cho Lan mẹ về trễ",
                "intent": "send-mess",
                "expected": {"RECEIVER": "lan", "MESSAGE": "mẹ về trễ"},
                "min_confidence": 0.8
            },
            {
                "input": "Gửi tới bố con đang ở ngoài",
                "intent": "send-mess",
                "expected": {"RECEIVER": "bố", "MESSAGE": "con đang ở ngoài"},
                "min_confidence": 0.8
            },
            # Case D: có platform
            {
                "input": "Nhắn cho anh Trường qua Zalo là con đến",
                "intent": "send-mess",
                "expected": {"RECEIVER": "anh trường", "MESSAGE": "con đến", "PLATFORM": "zalo"},
                "min_confidence": 1.0
            },
            {
                "input": "Gửi bằng SMS tới số này bác khỏe không",
                "intent": "send-mess",
                "expected": {"RECEIVER": "số này", "MESSAGE": "bác khỏe không", "PLATFORM": "sms"},
                "min_confidence": 0.8
            },
            # Edge cases
            {
                "input": "Nhắn giúp mẹ là nhớ khóa cửa nhé",
                "intent": "send-mess",
                "expected": {"RECEIVER": "mẹ", "MESSAGE": "nhớ khóa cửa nhé"},
                "min_confidence": 1.0
            },
            {
                "input": "Gửi tin nhắn cho Lan nói tối nay không ăn cơm",
                "intent": "send-mess",
                "expected": {"RECEIVER": "lan", "MESSAGE": "tối nay không ăn cơm"},
                "min_confidence": 0.8
            },
            {
                "input": "Nhắn cho chị Mai hôm nay mẹ về muộn",
                "intent": "send-mess",
                "expected": {"RECEIVER": "chị mai", "MESSAGE": "hôm nay mẹ về muộn"},
                "min_confidence": 0.8
            },
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




