package com.example.Auto_BE.dto.response;

import com.example.Auto_BE.entity.enums.EGender;
import lombok.Data;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
public class SupervisorUserResponse {
    private Long id;
    private String email;
    private String fullName;
    private LocalDate dateOfBirth;
    private EGender gender;
    private String phoneNumber;
    private String address;
    private String avatar;
    private String organization;
    private String licenseNumber;
    private String specialization;
    private Boolean isActive;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
