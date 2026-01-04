# 🧪 TESTING ROADMAP - PRIORITY ACTION PLAN

**Mục tiêu:** Nâng Testing Coverage từ 40% → 80%+  
**Thời gian:** 1-2 tuần  
**Priority:** HIGH

---

## 📋 PHASE 1: SETUP TEST INFRASTRUCTURE (2 ngày)

### Day 1: Setup pytest và test structure

```bash
# 1. Install testing dependencies
pip install pytest pytest-cov pytest-asyncio httpx

# 2. Create test directory structure
mkdir -p tests/unit tests/integration tests/fixtures tests/reports
touch tests/__init__.py
touch tests/conftest.py
```

**Files to create:**

```python
# tests/conftest.py
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

@pytest.fixture
def sample_texts():
    """Sample test texts for all intents"""
    return {
        "send-mess": [
            "gửi cho chị Mai nói tối nay con bận",
            "nhắn tin với cháu Hạnh chiều bà qua nhé qua zalo",
            "gửi tin nhắn cho mẹ hỏi có cần mua gì không",
        ],
        "call": [
            "gọi điện cho bố",
            "gọi cho bác sĩ Hùng",
            "call anh Tuấn",
        ],
        "set-alarm": [
            "đặt báo thức 7 giờ sáng mai",
            "báo thức 6 rưỡi sáng",
            "đặt alarm 8 giờ tối",
        ],
        "control-device": [
            "bật đèn flash",
            "tắt wifi",
            "mở bluetooth",
        ],
        "search-youtube": [
            "tìm kiếm nhạc trên youtube",
            "tìm video hướng dẫn nấu ăn trên youtube",
        ],
    }

@pytest.fixture
def entity_extractor():
    """Entity extractor instance"""
    from src.inference.engines.entity_extractor import EntityExtractor
    return EntityExtractor()

@pytest.fixture
def hybrid_system():
    """Hybrid system instance"""
    from core.hybrid_system import ModelFirstHybridSystem
    return ModelFirstHybridSystem()
```

### Day 2: Create test fixtures

```python
# tests/fixtures/test_cases.json
{
  "message_receiver_cases": [
    {
      "text": "gửi cho chị Mai nói tối nay con bận",
      "expected": {
        "RECEIVER": "chị Mai",
        "MESSAGE": "tối nay con bận",
        "PLATFORM": "sms"
      }
    },
    {
      "text": "nhắn tin với cháu Hạnh chiều bà qua nhé qua zalo",
      "expected": {
        "RECEIVER": "cháu Hạnh",
        "MESSAGE": "chiều bà qua nhé",
        "PLATFORM": "zalo"
      }
    }
  ],
  "alarm_cases": [
    {
      "text": "đặt báo thức 7 giờ sáng mai",
      "expected": {
        "TIME": "07:00",
        "DATE": "tomorrow",
        "TIMESTAMP": "ISO format"
      }
    },
    {
      "text": "báo thức 6 rưỡi sáng",
      "expected": {
        "TIME": "06:30"
      }
    }
  ],
  "device_cases": [
    {
      "text": "bật đèn flash",
      "expected": {
        "DEVICE": "flash",
        "ACTION": "ON"
      }
    },
    {
      "text": "tắt wifi",
      "expected": {
        "DEVICE": "wifi",
        "ACTION": "OFF"
      }
    }
  ]
}
```

---

## 📋 PHASE 2: UNIT TESTS (3-4 ngày)

### Test 1: MESSAGE/RECEIVER Extractor (Priority: CRITICAL)

```python
# tests/unit/test_message_receiver_extractor.py

import pytest
from src.inference.engines.entity_extractor import EntityExtractor

class TestMessageReceiverExtractor:
    """Test MESSAGE and RECEIVER extraction"""
    
    @pytest.fixture
    def extractor(self):
        return EntityExtractor()
    
    def test_case_a_gui_cho_x_noi_y(self, extractor):
        """Test Case A: gửi cho X nói/rằng/là Y"""
        text = "gửi cho chị Mai nói tối nay con bận"
        result = extractor.extract_message_receiver(text)
        
        assert result["RECEIVER"] == "chị Mai"
        assert "tối nay con bận" in result["MESSAGE"]
        assert result["PLATFORM"] in ["sms", "zalo"]
    
    def test_case_b_gui_x_noidung_y(self, extractor):
        """Test Case B: gửi X nội dung Y"""
        text = "gửi mẹ nội dung ở nhà đừng nấu cơm"
        result = extractor.extract_message_receiver(text)
        
        assert result["RECEIVER"] == "mẹ"
        assert "ở nhà đừng nấu cơm" in result["MESSAGE"]
    
    def test_case_c_nhan_x_noi_y(self, extractor):
        """Test Case C: nhắn X nói Y"""
        text = "nhắn bố nói tối về muộn"
        result = extractor.extract_message_receiver(text)
        
        assert result["RECEIVER"] == "bố"
        assert "tối về muộn" in result["MESSAGE"]
    
    def test_case_d_bao_x_rang_y(self, extractor):
        """Test Case D: báo X rằng Y"""
        text = "báo anh Tuấn rằng ngày mai họp"
        result = extractor.extract_message_receiver(text)
        
        assert result["RECEIVER"] == "anh Tuấn"
        assert "ngày mai họp" in result["MESSAGE"]
    
    def test_multi_token_receiver(self, extractor):
        """Test multi-token receiver (chị Mai, cô Hương)"""
        test_cases = [
            ("gửi cho chị Mai nói test", "chị Mai"),
            ("nhắn cô Hương hỏi test", "cô Hương"),
            ("gửi bác Tám báo test", "bác Tám"),
            ("nhắn cháu Hạnh nói test", "cháu Hạnh"),
        ]
        
        for text, expected_receiver in test_cases:
            result = extractor.extract_message_receiver(text)
            assert result["RECEIVER"] == expected_receiver, f"Failed for: {text}"
    
    def test_negative_case_nhan_tin(self, extractor):
        """Test that 'nhắn tin' is not extracted as receiver"""
        text = "nhắn tin với bạn nói test"
        result = extractor.extract_message_receiver(text)
        
        assert result["RECEIVER"] != "Tin"
        assert result["RECEIVER"] == "bạn"
    
    def test_platform_detection(self, extractor):
        """Test platform extraction and cleanup from MESSAGE"""
        text = "gửi cho mẹ nói test qua zalo"
        result = extractor.extract_message_receiver(text)
        
        assert result["PLATFORM"] == "zalo"
        assert "zalo" not in result["MESSAGE"].lower()
    
    @pytest.mark.parametrize("text,expected_receiver,expected_message", [
        ("gửi cho chị Mai nói tối nay con bận", "chị Mai", "tối nay con bận"),
        ("nhắn mẹ hỏi ăn gì", "mẹ", "ăn gì"),
        ("báo bố rằng về muộn", "bố", "về muộn"),
    ])
    def test_parametrized_cases(self, extractor, text, expected_receiver, expected_message):
        """Parametrized test for multiple cases"""
        result = extractor.extract_message_receiver(text)
        assert result["RECEIVER"] == expected_receiver
        assert expected_message in result["MESSAGE"]
```

### Test 2: ALARM Extractor

```python
# tests/unit/test_alarm_extractor.py

import pytest
from src.inference.engines.entity_extractor import EntityExtractor

class TestAlarmExtractor:
    """Test ALARM TIME/DATE extraction"""
    
    @pytest.fixture
    def extractor(self):
        return EntityExtractor()
    
    def test_time_extraction_numeric(self, extractor):
        """Test numeric time extraction (7 giờ, 7h30)"""
        test_cases = [
            ("đặt báo thức 7 giờ", "07:00"),
            ("báo thức 7h30", "07:30"),
            ("alarm 6 rưỡi", "06:30"),
            ("đặt báo thức 8 giờ 15", "08:15"),
        ]
        
        for text, expected_time in test_cases:
            result = extractor.extract_alarm_time_date(text)
            assert result.get("TIME") == expected_time, f"Failed for: {text}"
    
    def test_time_extraction_words(self, extractor):
        """Test word-based time (bảy giờ, sáu rưỡi)"""
        test_cases = [
            ("đặt báo thức bảy giờ", "07:00"),
            ("báo thức sáu rưỡi", "06:30"),
            ("alarm tám giờ", "08:00"),
        ]
        
        for text, expected_time in test_cases:
            result = extractor.extract_alarm_time_date(text)
            assert result.get("TIME") == expected_time, f"Failed for: {text}"
    
    def test_date_extraction_relative(self, extractor):
        """Test relative date (mai, hôm nay)"""
        test_cases = [
            ("báo thức 7 giờ sáng mai", "tomorrow"),
            ("đặt alarm 8 giờ hôm nay", "today"),
        ]
        
        for text, expected_date_type in test_cases:
            result = extractor.extract_alarm_time_date(text)
            assert "DATE" in result or "TIMESTAMP" in result
    
    def test_date_extraction_weekday(self, extractor):
        """Test weekday extraction (thứ 2, thứ 7, chủ nhật)"""
        test_cases = [
            "đặt báo thức 7 giờ thứ 2",
            "alarm 8 giờ thứ 7",
            "báo thức 6 giờ chủ nhật",
        ]
        
        for text in test_cases:
            result = extractor.extract_alarm_time_date(text)
            assert "DATE" in result or "DAYS_OF_WEEK" in result
    
    def test_timestamp_normalization(self, extractor):
        """Test TIMESTAMP ISO format"""
        text = "đặt báo thức 7 giờ sáng mai"
        result = extractor.extract_alarm_time_date(text)
        
        if "TIMESTAMP" in result:
            # Should be ISO format: YYYY-MM-DDTHH:MM:SS
            timestamp = result["TIMESTAMP"]
            assert "T" in timestamp
            assert len(timestamp) >= 19
```

### Test 3: DEVICE Extractor

```python
# tests/unit/test_device_extractor.py

import pytest
from src.inference.engines.entity_extractor import EntityExtractor

class TestDeviceExtractor:
    """Test DEVICE control extraction"""
    
    @pytest.fixture
    def extractor(self):
        return EntityExtractor()
    
    @pytest.mark.parametrize("text,expected_device,expected_action", [
        ("bật đèn flash", "flash", "ON"),
        ("tắt wifi", "wifi", "OFF"),
        ("mở bluetooth", "bluetooth", "ON"),
        ("tắt bluetooth", "bluetooth", "OFF"),
        ("bật mobile data", "mobile_data", "ON"),
        ("tăng âm lượng", "volume", "ON"),
        ("giảm âm lượng", "volume", "OFF"),
        ("tăng độ sáng", "brightness", "ON"),
    ])
    def test_device_action_extraction(self, extractor, text, expected_device, expected_action):
        """Test DEVICE and ACTION extraction"""
        result = extractor.extract_device_control(text)
        
        assert result.get("DEVICE") == expected_device, f"Failed device for: {text}"
        assert result.get("ACTION") == expected_action, f"Failed action for: {text}"
    
    def test_invalid_device(self, extractor):
        """Test that non-whitelisted devices are not extracted"""
        text = "bật tivi"  # tivi not in whitelist
        result = extractor.extract_device_control(text)
        
        assert result.get("DEVICE") is None
```

### Test 4: PLATFORM Extractor

```python
# tests/unit/test_platform_extractor.py

import pytest
from src.inference.engines.entity_extractor import EntityExtractor

class TestPlatformExtractor:
    """Test PLATFORM extraction"""
    
    @pytest.fixture
    def extractor(self):
        return EntityExtractor()
    
    @pytest.mark.parametrize("text,expected_platform", [
        ("gửi tin nhắn qua zalo", "zalo"),
        ("nhắn tin qua messenger", "messenger"),
        ("gửi qua facebook", "facebook"),
        ("nhắn qua viber", "viber"),
    ])
    def test_platform_extraction(self, extractor, text, expected_platform):
        """Test platform keyword extraction"""
        result = extractor._extract_platform(text)
        assert result == expected_platform
    
    def test_platform_cleanup_from_message(self, extractor):
        """Test platform removal from MESSAGE"""
        text = "gửi cho mẹ nói test qua zalo"
        result = extractor.extract_message_receiver(text)
        
        assert result.get("PLATFORM") == "zalo"
        assert "zalo" not in result["MESSAGE"].lower()
```

### Test 5: QUERY Extractor

```python
# tests/unit/test_query_extractor.py

import pytest
from src.inference.engines.entity_extractor import EntityExtractor

class TestQueryExtractor:
    """Test QUERY extraction for search intents"""
    
    @pytest.fixture
    def extractor(self):
        return EntityExtractor()
    
    @pytest.mark.parametrize("text,expected_query", [
        ("tìm kiếm nhà hàng gần đây", "nhà hàng gần đây"),
        ("tra cứu thời tiết hà nội", "thời tiết hà nội"),
        ("search video nấu ăn", "video nấu ăn"),
        ("tìm nhạc trên youtube", "nhạc"),
    ])
    def test_query_extraction(self, extractor, text, expected_query):
        """Test QUERY extraction with trigger removal"""
        # Assuming there's an extract_query method
        # This is a placeholder - adjust to actual method
        pass
```

---

## 📋 PHASE 3: INTEGRATION TESTS (2-3 ngày)

### Test 1: Hybrid System End-to-End

```python
# tests/integration/test_hybrid_system.py

import pytest
from core.hybrid_system import ModelFirstHybridSystem

class TestHybridSystemIntegration:
    """Integration tests for hybrid system"""
    
    @pytest.fixture(scope="class")
    def system(self):
        """Load system once for all tests"""
        return ModelFirstHybridSystem()
    
    def test_send_mess_intent(self, system):
        """Test send-mess full pipeline"""
        text = "gửi cho chị Mai nói tối nay con bận"
        result = system.process(text)
        
        assert result["intent"] == "send-mess"
        assert result["confidence"] > 0.5
        assert "RECEIVER" in result["entities"]
        assert "MESSAGE" in result["entities"]
        assert result["entity_clarity_score"] > 0.8
    
    def test_call_intent(self, system):
        """Test call full pipeline"""
        text = "gọi điện cho bố"
        result = system.process(text)
        
        assert result["intent"] == "call"
        assert "RECEIVER" in result["entities"]
    
    def test_set_alarm_intent(self, system):
        """Test set-alarm full pipeline"""
        text = "đặt báo thức 7 giờ sáng mai"
        result = system.process(text)
        
        assert result["intent"] == "set-alarm"
        assert "TIME" in result["entities"]
    
    def test_control_device_intent(self, system):
        """Test control-device full pipeline"""
        text = "bật đèn flash"
        result = system.process(text)
        
        assert result["intent"] == "control-device"
        assert "DEVICE" in result["entities"]
        assert "ACTION" in result["entities"]
    
    @pytest.mark.parametrize("text,expected_intent", [
        ("gọi điện cho mẹ", "call"),
        ("nhắn tin cho bạn", "send-mess"),
        ("đặt báo thức 7 giờ", "set-alarm"),
        ("bật wifi", "control-device"),
        ("tìm kiếm trên youtube", "search-youtube"),
    ])
    def test_intent_classification_accuracy(self, system, text, expected_intent):
        """Test intent classification for multiple cases"""
        result = system.process(text)
        assert result["intent"] == expected_intent
```

### Test 2: API Endpoints

```python
# tests/integration/test_api_endpoints.py

import pytest
from fastapi.testclient import TestClient
from api.server import app

class TestAPIEndpoints:
    """Test API endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test /health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_predict_endpoint(self, client):
        """Test /predict endpoint"""
        request = {
            "text": "gọi điện cho mẹ",
            "context": None
        }
        response = client.post("/predict", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert "intent" in data
        assert "entities" in data
        assert "confidence" in data
        assert "entity_clarity_score" in data
    
    def test_stats_endpoint(self, client):
        """Test /stats endpoint"""
        response = client.get("/stats")
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_multiple_requests(self, client):
        """Test multiple concurrent requests"""
        test_texts = [
            "gọi điện cho mẹ",
            "nhắn tin cho bạn",
            "đặt báo thức 7 giờ",
        ]
        
        for text in test_texts:
            response = client.post("/predict", json={"text": text})
            assert response.status_code == 200
```

---

## 📋 PHASE 4: TEST EXECUTION & REPORTING (1 ngày)

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov=core --cov-report=html --cov-report=term

# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v

# Run with markers
pytest -m "slow" -v  # Only slow tests
pytest -m "not slow" -v  # Skip slow tests

# Generate HTML report
pytest tests/ --html=tests/reports/test_report.html --self-contained-html
```

---

## 📊 SUCCESS CRITERIA

- [ ] **Unit test coverage: 80%+**
  - [ ] Entity extractors: 100% coverage
  - [ ] Core hybrid logic: 90%+ coverage
  - [ ] Utility functions: 80%+ coverage

- [ ] **Integration test coverage: 60%+**
  - [ ] API endpoints: 100% coverage
  - [ ] End-to-end pipelines: 80%+ coverage

- [ ] **Test pass rate: 95%+**
  - [ ] All critical tests pass
  - [ ] No flaky tests

- [ ] **Performance benchmarks**
  - [ ] Test execution time < 5 minutes
  - [ ] No memory leaks
  - [ ] API tests < 1s per request

---

## 📝 DELIVERABLES

1. **Test Suite**
   - `tests/unit/` - 5 test files (200+ test cases)
   - `tests/integration/` - 2 test files (50+ test cases)
   - `tests/fixtures/` - Test data
   - `tests/conftest.py` - Shared fixtures

2. **Test Reports**
   - `tests/reports/coverage_report.html`
   - `tests/reports/test_results.html`
   - `tests/reports/performance_benchmarks.json`

3. **CI/CD Config**
   - `.github/workflows/test.yml`
   - `pytest.ini`
   - `.coveragerc`

---

## 🎯 TIMELINE

| Day | Task | Hours | Status |
|-----|------|-------|--------|
| 1-2 | Setup infrastructure | 16h | [ ] |
| 3-6 | Unit tests (5 extractors) | 32h | [ ] |
| 7-9 | Integration tests | 24h | [ ] |
| 10 | Test execution & reporting | 8h | [ ] |

**Total:** ~80 hours (10 days full-time, or 2 weeks part-time)

---

**Priority:** HIGH  
**Start Date:** TBD  
**Owner:** TBD  
**Status:** 📝 PLANNED


