package com.example.Auto_BE.dto.response;

import com.example.Auto_BE.entity.enums.EBloodType;
import com.example.Auto_BE.entity.enums.EGender;
import lombok.*;

import java.time.LocalDate;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ProfileResponse {
    private Long id;
    private String fullName;
    private String email;
    private LocalDate dateOfBirth;
    private EGender gender;
    private String phoneNumber;
    private String address;
    private String avatar;
    private Boolean isActive;
    
    // Role: ELDER, SUPERVISOR, USER
    private String role;
    
    // Elder-specific fields (chỉ có khi role = ELDER)
    private EBloodType bloodType;
    private Double height;
    private Double weight;
    
    // Supervisor-specific fields (chỉ có khi role = SUPERVISOR)
    private String occupation; // Nghề nghiệp
    private String workplace; // Nơi làm việc
}