package com.example.Auto_BE.dto.request;

import com.example.Auto_BE.entity.enums.ETypeMedication;
import jakarta.validation.constraints.*;

import lombok.*;

import java.util.List;

@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class MedicationReminderCreateRequest {
    @NotBlank(message = "Tên thuốc không được để trống")
    private String name;

    @Size(max = 1000, message = "Mô tả thuốc tối đa 1000 ký tự")
    private String description;

    @NotNull(message = "Loại thuốc không được để trống")
    private ETypeMedication type;

    @NotEmpty(message = "Phải có ít nhất 1 giờ uống thuốc")
    private List<
            @Pattern(regexp = "^([01]\\d|2[0-3]):[0-5]\\d$", message = "Giờ uống thuốc không đúng định dạng (HH:mm)")
                    String
            > reminderTimes;

    // Ngày trong tuần: '1111111' = hàng ngày, '11111100' = T2-T6, '1000001' = T2&CN
    @NotBlank(message = "Phải chọn ít nhất 1 ngày trong tuần")
    @Pattern(regexp = "^[01]{7}$", message = "Ngày trong tuần không đúng định dạng (7 ký tự 0 hoặc 1)")
    private String daysOfWeek;
}