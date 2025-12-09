package com.example.Auto_BE.entity;

import com.example.Auto_BE.entity.enums.ENotificationStatus;
import com.example.Auto_BE.entity.enums.ENotificationType;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;

import java.time.LocalDateTime;

@Entity
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Accessors(chain = true)
@Table(name = "notifications")
public class Notifications extends BaseEntity{
    
    @Column(name = "notification_type", nullable = false)
    @Enumerated(EnumType.STRING)
    private ENotificationType notificationType; // Loại thông báo
    
    @Column(name = "title", length = 255, nullable = false)
    private String title; // Tiêu đề thông báo

    @Column(name = "body", length = 1000)
    private String body; // Nội dung thông báo
    
    @Column(name = "status", nullable = false)
    @Enumerated(EnumType.STRING)
    private ENotificationStatus status = ENotificationStatus.SENT; // Trạng thái thông báo

    @Column(name = "is_read", nullable = false)
    private Boolean isRead = false; // Đã xem thông báo hay chưa

    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "user_id", nullable = false)
    private User user; // Người NHẬN thông báo (ElderUser HOẶC SupervisorUser)
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "related_elder_id")
    private ElderUser relatedElder; // Elder liên quan (nếu notification cho Supervisor về Elder cụ thể)
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "related_medication_log_id")
    private MedicationLog relatedMedicationLog; // Link đến MedicationLog (nếu là MEDICATION_REMINDER)
    
    @Column(name = "action_url", length = 500)
    private String actionUrl; // Deep link hoặc URL hành động (VD: "/medication-logs/123")
}