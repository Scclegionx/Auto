package com.example.Auto_BE.dto.request;

import com.example.Auto_BE.entity.enums.EBloodType;
import com.example.Auto_BE.entity.enums.EGender;
import jakarta.validation.constraints.Pattern;
import lombok.*;

import java.time.LocalDate;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UpdateProfileRequest {
    private String fullName;
    private LocalDate dateOfBirth;
    private EGender gender;
    @Pattern(regexp = "^[+]?[0-9]{10,15}$")
    private String phoneNumber;
    private String address;
    private EBloodType bloodType;
    private Double height;
    private Double weight;
}
    