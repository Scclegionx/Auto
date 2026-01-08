package com.example.Auto_BE.dto.request;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class UpdateUserSettingRequest {

    private Long userId;
    
    @NotBlank(message = "Setting key is required")
    @Size(max = 100, message = "Setting key too long")
    private String settingKey;
    
    @NotBlank(message = "Value is required")
    @Size(max = 255, message = "Value too long")
    private String value;
}

