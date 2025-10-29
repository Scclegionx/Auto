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
    private LocalDateTime reminderTime; // Thời gian nhắc nhở, ví dụ: "2023-10-01T10:00:00"

    @Column(name = "title", length = 255)
    private String title; // Tiêu đề thông báo đã gửi

    @Column(name = "body", length = 1000)
    private String body; // Nội dung thông báo đã gửi

    @Column(name = "medication_count", nullable = false)
    private Integer medicationCount = 1; // Số lượng thuốc trong thông báo này

    @Column(name = "medication_ids", length = 500)
    private String medicationIds; // Danh sách ID thuốc cách nhau bởi dấu phẩy, ví dụ: "1,5,8"

    @Column(name = "medication_names", length = 1000)
    private String medicationNames; // Danh sách tên thuốc cách nhau bởi dấu phẩy, ví dụ: "Paracetamol,Vitamin C,Amoxicillin"

    @Column(name = "status", nullable = false)
    @Enumerated(EnumType.STRING)
    private ENotificationStatus status = ENotificationStatus.SENT; // Trạng thái thông báo, mặc định là SENT

    @Column(name = "is_read", nullable = false)
    private Boolean isRead = false; // Đã xem thông báo hay chưa, mặc định là chưa xem

    @Column(name = "last_sent_time")
    private LocalDateTime lastSentTime; // Thời gian gửi thông báo lần cuối, có thể null nếu chưa gửi lần nào

    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "user_id", nullable = false)
    private User user; // Người dùng sở hữu thông báo này
}