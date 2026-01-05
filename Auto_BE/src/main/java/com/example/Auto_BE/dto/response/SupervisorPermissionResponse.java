package com.example.Auto_BE.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SupervisorPermissionResponse {
    
    private Long supervisorId;
    private String supervisorName;
    private String supervisorEmail;
    
    private Long elderId;
    private String elderName;
    private String elderEmail;
    
    private Boolean canViewMedications;
    private Boolean canUpdateMedications;
    private Boolean isActive;
    
    private String relationshipStatus; // ACCEPTED, PENDING, REJECTED
}
