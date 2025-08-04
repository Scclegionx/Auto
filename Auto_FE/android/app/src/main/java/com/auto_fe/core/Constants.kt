package com.auto_fe.core

object Constants {
    
    // Service Actions
    const val ACTION_SEND_MESSAGE = "com.auto_fe.ACTION_SEND_MESSAGE"
    const val ACTION_MAKE_CALL = "com.auto_fe.ACTION_MAKE_CALL"
    const val ACTION_SEARCH_WEB = "com.auto_fe.ACTION_SEARCH_WEB"
    
    // Commands
    const val COMMAND_SEND_MESSAGE = "send-mes"
    const val COMMAND_MAKE_CALL = "make-call"
    const val COMMAND_SEARCH_WEB = "search-web"
    
    // Intent Extras
    object IntentExtras {
        const val COMMAND = "command"
        const val ENTITIES = "entities"
        const val VALUES = "values"
    }
    
    // Error Messages
    object ErrorMessages {
        const val CONTACT_NOT_FOUND = "Không tìm thấy liên hệ"
        const val PHONE_NUMBER_NOT_FOUND = "Không tìm thấy số điện thoại"
        const val PERMISSION_DENIED = "Quyền bị từ chối"
        const val SERVICE_NOT_AVAILABLE = "Dịch vụ không khả dụng"
    }
    
    // Success Messages
    object SuccessMessages {
        const val MESSAGE_SENT = "Tin nhắn đã được gửi"
        const val CALL_STARTED = "Cuộc gọi đã được bắt đầu"
        const val SEARCH_OPENED = "Tìm kiếm đã được mở"
    }
    
    // Search Engines
    object SearchEngines {
        const val GOOGLE = "google"
        const val BING = "bing"
        const val YAHOO = "yahoo"
    }
    
    // JSON Keys
    object JsonKeys {
        const val ENTITY = "ent"
        const val VALUE = "val"
        const val ENGINE = "engine"
        const val QUERY = "query"
        const val CONTACT = "contact"
        const val MESSAGE = "message"
    }
    
    // Test Data
    object TestData {
        const val TEST_CONTACT = "mom"
        const val TEST_MESSAGE = "con sắp về"
        const val TEST_ENTITIES = """{"ent": "mom"}"""
        const val TEST_VALUES = """{"val": "con sắp về"}"""
    }
    
    // Toast Messages
    object ToastMessages {
        const val STARTING_MESSAGE_TEST = "Bắt đầu test gửi tin nhắn..."
        const val MESSAGE_APP_OPENED = "Mở app tin nhắn để gửi cho"
        const val CANNOT_OPEN_MESSAGE_APP = "Không thể mở app tin nhắn"
        const val CONTACT_NOT_FOUND = "Không tìm thấy số điện thoại cho"
        const val INVALID_DATA = "Dữ liệu không hợp lệ"
        const val UNSUPPORTED_COMMAND = "Lệnh không được hỗ trợ"
        const val ERROR_PREFIX = "Lỗi:"
        const val MESSAGE_ERROR_PREFIX = "Lỗi gửi tin nhắn:"
        const val COMMAND_EXECUTION_ERROR = "Lỗi thực thi lệnh:"
    }
} 