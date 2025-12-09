package com.example.Auto_BE.entity;

import com.example.Auto_BE.entity.enums.EMedicationLogStatus;
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
@Table(name = "medication_logs")
public class MedicationLog extends BaseEntity {
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "elder_user_id", nullable = false)
    private ElderUser elderUser; // CHỈ Elder - người cần uống thuốc
    
    @Column(name = "reminder_time", nullable = false)
    private LocalDateTime reminderTime; // Thời gian dự kiến uống (09:00)
    
    @Column(name = "actual_taken_time")
    private LocalDateTime actualTakenTime; // Thời gian user XÁC NHẬN đã uống (09:15)
    
    @Column(name = "minutes_late")
    private Integer minutesLate; // Số phút chênh lệch (dương = trễ, âm = sớm)
    
    @Column(name = "status", nullable = false)
    @Enumerated(EnumType.STRING)
    private EMedicationLogStatus status = EMedicationLogStatus.PENDING; // PENDING, TAKEN, MISSED
    
    @Column(name = "medication_count", nullable = false)
    private Integer medicationCount = 1; // Số lượng thuốc trong log này
    
    @Column(name = "medication_ids", length = 500)
    private String medicationIds; // Danh sách ID thuốc: "1,5,8"
    
    @Column(name = "medication_names", length = 1000)
    private String medicationNames; // Danh sách tên thuốc: "Paracetamol, Vitamin C"
    
    @Column(name = "note", length = 500)
    private String note; // Ghi chú của elder khi xác nhận
    
    @Column(name = "fcm_sent", nullable = false)
    private Boolean fcmSent = false; // Đã gửi FCM notification chưa
    
    @Column(name = "fcm_sent_time")
    private LocalDateTime fcmSentTime; // Thời gian gửi FCM
}
