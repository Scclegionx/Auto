package com.example.Auto_BE.dto.request;

import com.example.Auto_BE.entity.enums.ETypeMedication;
import jakarta.validation.constraints.*;
import lombok.*;

import java.util.List;

/**
 * 💊 UNIFIED Request cho Individual Medication CRUD
 * 
 * ✅ Đồng bộ với MedicationReminderCreateRequest (dùng trong Prescription)
 * ✅ Cùng lưu vào bảng MedicationReminder
 * ✅ Hỗ trợ TIME-BASED scheduling
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class CreateMedicationRequest {

    @NotBlank(message = "Medication name is required")
    private String name; // Tên thuốc

    @Size(max = 1000, message = "Description too long")
    private String description; // Mô tả, hướng dẫn sử dụng

    @NotNull(message = "Medication type is required")
    private ETypeMedication type; // Loại thuốc: PILL, LIQUID, INJECTION, etc.

    @NotEmpty(message = "At least one reminder time is required")
    private List<
            @Pattern(regexp = "^([01]\\d|2[0-3]):[0-5]\\d$", message = "Invalid time format. Use HH:mm")
            String
    > reminderTimes; // Danh sách giờ nhắc: ["08:00", "14:00", "20:00"]

    // Ngày trong tuần: '1111111' = hàng ngày, '1111100' = T2-T6, '1000001' = T2&CN
    @NotBlank(message = "Days of week is required")
    @Pattern(regexp = "^[01]{7}$", message = "Invalid days format. Use 7 digits (0 or 1)")
    private String daysOfWeek; // "1111111" cho hàng ngày

    // Optional fields (không bắt buộc)
    private Long prescriptionId; // NULL nếu là thuốc riêng lẻ, có giá trị nếu thuộc đơn

    @Builder.Default
    private Boolean isActive = true; // Mặc định active
}