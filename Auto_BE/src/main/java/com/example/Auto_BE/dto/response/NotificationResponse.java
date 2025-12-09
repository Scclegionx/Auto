package com.example.Auto_BE.dto.response;

import com.example.Auto_BE.entity.enums.ENotificationStatus;
import com.example.Auto_BE.entity.enums.ENotificationType;
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
    
    // Notification type (MEDICATION_REMINDER, ELDER_MISSED_MEDICATION, etc.)
    private ENotificationType notificationType;
    
    // Notification content
    private String title;
    private String body;
    
    // Status
    private ENotificationStatus status;
    private Boolean isRead;
    
    // Action & Links
    private String actionUrl; // Deep link for navigation
    private Long relatedElderId; // For supervisor notifications about elder
    private Long relatedMedicationLogId; // Link to medication log
    
    // User info
    private Long userId;
    private String userEmail;
    
    // Timestamps
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
