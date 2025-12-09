package com.example.Auto_BE.entity.enums;

public enum EMedicationLogStatus {
    PENDING,    // Chưa đến giờ hoặc đang chờ xác nhận
    TAKEN,      // Đã uống (user đã confirm)
    MISSED,     // Bỏ lỡ (quá giờ >30 phút mà chưa confirm)
    SKIPPED     // Bỏ qua có chủ đích (optional)
}
