package com.example.Auto_BE.dto.request;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import lombok.*;

import java.util.List;

@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class PrescriptionCreateRequest {
    @NotBlank
    private String name;

    @NotBlank
    private String description;

    @NotBlank
    private String imageUrl;

    // Cho phép rỗng (đơn không có nhắc nhở)
    // Thêm @Valid để validate từng phần tử trong list
    @Valid
    private List<MedicationReminderCreateRequest> medicationReminders;
}
