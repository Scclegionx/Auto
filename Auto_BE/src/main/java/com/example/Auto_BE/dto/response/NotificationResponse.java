package com.example.Auto_BE.dto.response;

import com.example.Auto_BE.entity.enums.ENotificationStatus;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class NotificationResponse {
    private Long id;
    private LocalDateTime reminderTime;
    private LocalDateTime lastSentTime;
    private ENotificationStatus status;
    private Boolean isRead;
    
    // Thông tin về thông báo
    private String title;
    private String body;
    
    // Thông tin về thuốc
    private Integer medicationCount;
    private String medicationIds; // "1,5,8"
    private String medicationNames; // "Paracetamol, Vitamin C, Amoxicillin"
    
    // Thông tin user
    private Long userId;
    private String userEmail;
}
