package com.example.Auto_BE.dto.request;

import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class ElderSupervisorRequest {
    
    @NotNull(message = "Elder user ID is required")
    private Long elderUserId;
    
    @NotNull(message = "Supervisor user ID is required")
    private Long supervisorUserId;
    
    private Boolean canViewPrescription = false;
    
    private Boolean canUpdatePrescription = false;
    
    private String note;
}
