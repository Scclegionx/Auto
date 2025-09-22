package com.auto_fe.auto_fe.utils

object Config {
    // Server NLP Configuration
    const val NLP_SERVER_URL = "http://192.168.100.91:8000/infer" 
    const val NLP_TIMEOUT_SECONDS = 30L
    
    // Audio Configuration
    const val AUDIO_SAMPLE_RATE = 44100
    const val AUDIO_CHANNEL_CONFIG = android.media.AudioFormat.CHANNEL_IN_MONO
    const val AUDIO_ENCODING = android.media.AudioFormat.ENCODING_PCM_16BIT
    
    // Speech Recognition Configuration
    const val SPEECH_LANGUAGE = "vi-VN"
    const val SPEECH_TIMEOUT_MS = 10000L
    
    // Floating Window Configuration
    const val FLOATING_WINDOW_X = 0
    const val FLOATING_WINDOW_Y = 100
    
    // Commands
    object Commands {
        const val SMS = "sms"
        const val PHONE = "call"
        const val MESSAGE = "nhắn tin"
        const val CALL = "gọi"
    }
    
    // Error Messages
    object Messages {
        const val PERMISSION_DENIED = "Cần cấp quyền để sử dụng ứng dụng"
        const val OVERLAY_PERMISSION_DENIED = "Cần cấp quyền hiển thị trên các ứng dụng khác"
        const val NETWORK_ERROR = "Lỗi kết nối mạng"
        const val NLP_ERROR = "Lỗi xử lý ngôn ngữ tự nhiên"
        const val AUDIO_ERROR = "Lỗi ghi âm"
        const val COMMAND_CANCELLED = "Lệnh đã bị hủy"
        const val COMMAND_EXECUTED = "Lệnh đã được thực hiện"
    }
}
