package com.example.Auto_BE.dto.request;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import jakarta.validation.constraints.NotNull;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class SendRelationshipRequestDTO {
    
    @NotNull(message = "Target user ID không được để trống")
    private Long targetUserId;
    
    private String message; // Lời nhắn khi gửi request
}
