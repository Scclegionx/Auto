package com.auto_fe.auto_fe.base

/**
 * AutomationStates - Định nghĩa các trạng thái cơ bản cho workflow
 * Tất cả automation đều đi qua các bước này
 */
sealed class AutomationState {
    /**
     * Bước 1: App đang nói (chào hỏi hoặc thông báo)
     * @param text Text cần nói
     */
    data class Speaking(val text: String) : AutomationState()
    
    /**
     * Bước 2: App đang lắng nghe người dùng nói
     */
    object Listening : AutomationState()
    
    /**
     * Bước 3: App đang xử lý logic nghiệp vụ
     * @param rawInput Dữ liệu đầu vào từ người dùng (text từ speech recognition)
     */
    data class Processing(val rawInput: String) : AutomationState()
    
    /**
     * Bước 4: Hoàn thành thành công
     * @param message Thông báo kết quả
     */
    data class Success(val message: String) : AutomationState()
    
    /**
     * Bước 5: Lỗi xảy ra
     * @param message Thông báo lỗi
     */
    data class Error(val message: String) : AutomationState()
}

