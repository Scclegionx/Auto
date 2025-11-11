package com.example.Auto_BE.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.Instant;
import java.time.LocalDateTime;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class EmergencyContactResponse {
    
    private Long id;
    private String name;
    private String phoneNumber;
    private String address;
    private String relationship;
    private String note;
    private Long userId;
    private Instant createdAt;
    private Instant updatedAt;
}
