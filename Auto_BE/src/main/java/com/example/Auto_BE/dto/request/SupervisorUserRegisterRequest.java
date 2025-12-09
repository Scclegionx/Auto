package com.example.Auto_BE.dto.request;

import com.example.Auto_BE.entity.enums.EGender;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

import java.time.LocalDate;

@Data
public class SupervisorUserRegisterRequest {
    
    @NotBlank(message = "Email is required")
    @Email(message = "Email is invalid")
    private String email;
    
    @NotBlank(message = "Password is required")
    private String password;
    
    @NotBlank(message = "Full name is required")
    private String fullName;
    
    private LocalDate dateOfBirth;
    
    private EGender gender;
    
    private String phoneNumber;
    
    private String address;
    
    private String organization;
    
    private String licenseNumber;
    
    private String specialization;
}
