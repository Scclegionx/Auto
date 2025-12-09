package com.example.Auto_BE.dto.response;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class ElderSupervisorResponse {
    private Long id;
    private Long elderUserId;
    private String elderUserName;
    private Long supervisorUserId;
    private String supervisorUserName;
    private Boolean canViewPrescription;
    private Boolean canUpdatePrescription;
    private Boolean isActive;
    private String note;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
