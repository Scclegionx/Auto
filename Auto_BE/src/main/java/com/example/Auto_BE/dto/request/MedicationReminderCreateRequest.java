package com.example.Auto_BE.dto.request;

import com.example.Auto_BE.entity.enums.ETypeMedication;
import jakarta.validation.constraints.*;

import lombok.*;

import java.util.List;

@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class MedicationReminderCreateRequest {
    @NotBlank
    private String name;

    @Size(max = 1000)
    private String description;

    @NotNull
    private ETypeMedication type;

    @NotEmpty
    private List<
            @Pattern(regexp = "^([01]\\d|2[0-3]):[0-5]\\d$")
                    String
            > reminderTimes;

    // Ngày trong tuần: '1111111' = hàng ngày, '11111100' = T2-T6, '1000001' = T2&CN
    @NotBlank
    @Pattern(regexp = "^[01]{7}$")
    private String daysOfWeek;
}