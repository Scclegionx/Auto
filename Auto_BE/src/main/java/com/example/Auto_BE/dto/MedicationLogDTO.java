package com.example.Auto_BE.dto;

import com.example.Auto_BE.entity.enums.EMedicationLogStatus;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;
import java.time.LocalDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MedicationLogDTO {
    private Long id;
    private Long elderUserId;
    private String medicationIds;
    private String medicationNames;
    private Integer medicationCount;
    private LocalDateTime reminderTime;
    private LocalDateTime actualTakenTime;
    private EMedicationLogStatus status;
    private Integer minutesLate;
    private String note;
    private Boolean fcmSent;
    private LocalDateTime fcmSentTime;
    private Instant createdAt;
    private Instant updatedAt;
}
