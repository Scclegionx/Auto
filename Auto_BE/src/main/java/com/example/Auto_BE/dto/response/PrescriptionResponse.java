package com.example.Auto_BE.dto.response;

import lombok.*;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class PrescriptionResponse {
    private Long id;
    private String name;
    private String description;
    private String imageUrl;
    private Boolean isActive;
    private Long userId;

    // Grouped medications (name + reminderTimes array)
    private List<MedicationResponse> medications;
    
    // Legacy field (deprecated, kept for backward compatibility)
    @Deprecated
    private List<MedicationReminderResponse> medicationReminders;
}