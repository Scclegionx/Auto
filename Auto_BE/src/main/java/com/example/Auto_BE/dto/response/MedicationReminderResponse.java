package com.example.Auto_BE.dto.response;

import com.example.Auto_BE.entity.enums.ETypeMedication;
import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class MedicationReminderResponse {
    private Long id;
    private String name;
    private String description;
    private ETypeMedication type;
    private String reminderTime; // "HH:mm"
    private String daysOfWeek;   // "1111111" = hàng ngày, "0111110" = T2-T6
    private Boolean isActive;

    private Long prescriptionId;
    private Long userId;
}
