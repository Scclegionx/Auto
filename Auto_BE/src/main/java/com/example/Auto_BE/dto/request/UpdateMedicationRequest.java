package com.example.Auto_BE.dto.request;

import com.example.Auto_BE.entity.enums.ETypeMedication;
import jakarta.validation.constraints.*;
import lombok.*;

import java.util.List;

/**
 * 💊 UNIFIED Update Request cho Individual Medication
 * 
 * ✅ Đồng bộ với MedicationReminderCreateRequest structure
 * ✅ Tất cả fields optional (chỉ update fields được gửi)
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class UpdateMedicationRequest {

    private String name; // Tên thuốc

    @Size(max = 1000, message = "Description too long")
    private String description; // Mô tả

    private ETypeMedication type; // Loại thuốc

    private List<
            @Pattern(regexp = "^([01]\\d|2[0-3]):[0-5]\\d$", message = "Invalid time format. Use HH:mm")
            String
    > reminderTimes; // Giờ nhắc: ["08:00", "14:00"]

    @Pattern(regexp = "^[01]{7}$", message = "Invalid days format. Use 7 digits")
    private String daysOfWeek; // "1111111"

    private Boolean isActive; // Active status
}