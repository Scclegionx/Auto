package com.example.Auto_BE.dto.request;

import com.example.Auto_BE.entity.enums.EBloodType;
import com.example.Auto_BE.entity.enums.EGender;
import jakarta.validation.constraints.*;
import lombok.*;

import java.time.LocalDate;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UpdateProfileRequest {
    @Size(min = 2, max = 100, message = "Họ tên phải từ 2 đến 100 ký tự")
    private String fullName;
    
    @Past(message = "Ngày sinh phải là ngày trong quá khứ")
    private LocalDate dateOfBirth;
    
    private EGender gender;
    
    @Pattern(regexp = "^[+]?[0-9]{10,15}$", message = "Số điện thoại không hợp lệ (10-15 chữ số)")
    private String phoneNumber;
    
    @Size(max = 200, message = "Địa chỉ tối đa 200 ký tự")
    private String address;
    
    private EBloodType bloodType;
    
    @DecimalMin(value = "50.0", message = "Chiều cao phải lớn hơn 50cm")
    @DecimalMax(value = "250.0", message = "Chiều cao phải nhỏ hơn 250cm")
    private Double height;
    
    @DecimalMin(value = "10.0", message = "Cân nặng phải lớn hơn 10kg")
    @DecimalMax(value = "300.0", message = "Cân nặng phải nhỏ hơn 300kg")
    private Double weight;
}
