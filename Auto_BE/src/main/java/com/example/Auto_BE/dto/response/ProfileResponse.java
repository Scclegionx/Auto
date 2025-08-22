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
    private EBloodType bloodType;
    private Double height;
    private Double weight;
    private String avatar;
    private Boolean isActive;
}
