package com.example.Auto_BE.dto.response;

import com.example.Auto_BE.entity.enums.EBloodType;
import com.example.Auto_BE.entity.enums.EGender;
import lombok.Data;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
public class ElderUserResponse {
    private Long id;
    private String email;
    private String fullName;
    private LocalDate dateOfBirth;
    private EGender gender;
    private String phoneNumber;
    private String address;
    private EBloodType bloodType;
    private Double height;
    private Double weight;
    private String avatar;
    private Boolean isActive;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
