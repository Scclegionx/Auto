package com.example.Auto_BE.entity;

import com.example.Auto_BE.entity.enums.ENotificationStatus;
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
    @Column(name = "reminder_time", nullable = false)
    private LocalDateTime reminderTime; // Dạng chuỗi, ví dụ: "2023-10-01T10:00:00"

    @Column(name = "status", nullable = false)
    @Enumerated(EnumType.STRING)
    private ENotificationStatus status= ENotificationStatus.PENDING; // Trạng thái thông báo, mặc định là PENDING

    @Column(name = "retry_count", nullable = false)
    private Integer retryCount = 0; // Số lần thử gửi thông báo, mặc định là 0

    @Column(name = "last_sent_time")
    private LocalDateTime lastSentTime; // Thời gian gửi thông báo lần cuối, có thể null nếu chưa gửi lần nào

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "medication_reminder_id", nullable = false)
    private MedicationReminder medicationReminder; // Nhắc nhở thuốc liên quan đến thông báo này

    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "user_id", nullable = false)
    private User user; // Người dùng sở hữu thông báo này


}