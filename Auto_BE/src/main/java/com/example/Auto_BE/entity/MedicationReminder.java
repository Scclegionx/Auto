package com.example.Auto_BE.entity;

import com.example.Auto_BE.entity.enums.ETypeMedication;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;

import java.util.List;

@Entity
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Accessors(chain = true)
@Table(name = "medication_reminders")
public class MedicationReminder extends BaseEntity {
    @Column(name = "name", nullable = false, length = 255)
    private String name; // Tên của nhắc nhở thuốc

    @Column(name = "description", length = 1000)
    private String description; // Mô tả chi tiết về nhắc nhở thuốc

    @Column(name = "type", nullable = false, length = 50)
    @Enumerated(EnumType.STRING)
    private ETypeMedication type;

    @Column(name = "reminder_time", nullable = false)
    private String reminderTime; // Thời gian nhắc nhở, định dạng "HH:mm"

    @Column(name = "days_of_week", nullable = false)
    private String daysOfWeek; // Mã tuần, ví dụ '1111111' cho hàng ngày, '0111110' cho T2-T6 (0 = không, 1 = có)

    @Column(name = "is_active", nullable = false)
    private Boolean isActive = true; // Trạng thái hoạt động của nhắc nhở thuốc

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user; // Người dùng sở hữu nhắc nhở thuốc này

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "prescription_id", nullable = true)//null khi thuốc mua ngoài
    private Prescriptions prescription; // Đơn thuốc liên quan đến nhắc nhở thuốc này


}