package com.example.Auto_BE.entity.enums;

public enum ENotificationType {
    // ===== Thông báo cho ELDER =====
    MEDICATION_REMINDER,           // Nhắc uống thuốc (link đến MedicationLog)
    
    // ===== Thông báo cho SUPERVISOR =====
    ELDER_MISSED_MEDICATION,       // Elder bỏ lỡ uống thuốc
    ELDER_LATE_MEDICATION,         // Elder uống trễ
    ELDER_ADHERENCE_LOW,           // Tỷ lệ tuân thủ thấp
    ELDER_HEALTH_ALERT,            // Cảnh báo sức khỏe
    
    // ===== Thông báo chung =====
    SYSTEM_ANNOUNCEMENT,           // Thông báo hệ thống
    RELATIONSHIP_REQUEST,          // Yêu cầu kết nối (Supervisor → Elder)
    RELATIONSHIP_ACCEPTED          // Chấp nhận kết nối
}
