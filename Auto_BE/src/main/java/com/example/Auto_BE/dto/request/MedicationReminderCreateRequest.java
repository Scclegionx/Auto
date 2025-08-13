package com.example.Auto_BE.dto.request;

import com.example.Auto_BE.entity.enums.ETypeMedication;
import jakarta.validation.constraints.*;

import lombok.*;

import java.util.List;

@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class MedicationReminderCreateRequest {
    @NotBlank(message = "Tên thuốc không được để trống")
    private String name;

    @Size(max = 1000, message = "Mô tả không được vượt quá 1000 ký tự")
    private String description;

    @NotNull(message = "Loại thuốc không được để trống")
    private ETypeMedication type;

    /** Ưu tiên dùng mảng này */
    @NotEmpty(message = "Cần ít nhất 1 giờ nhắc")
    private List<
                @Pattern(regexp = "^([01]\\d|2[0-3]):[0-5]\\d$", message = "Giờ phải có định dạng HH:mm")
                        String
                > reminderTimes;

    // Ngày trong tuần: '1111111' = hàng ngày, '0111110' = T2-T6, '1000001' = T2&CN
    @NotBlank(message = "Ngày trong tuần không được để trống")
    @Pattern(regexp = "^[01]{7}$", message = "Ngày trong tuần phải là chuỗi 7 ký tự 0/1 (T2,T3,T4,T5,T6,T7,CN)")
    private String daysOfWeek;
}
