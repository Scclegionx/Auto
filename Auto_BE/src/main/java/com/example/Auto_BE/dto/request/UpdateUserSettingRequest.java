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

    private Long userId; // null nếu là GUEST, có giá trị nếu đã đăng nhập
    
    @NotBlank(message = "Setting key is required")
    @Size(max = 100, message = "Setting key too long")
    private String settingKey; // "theme", "font_size", "voice_support"
    
    @NotBlank(message = "Value is required")
    @Size(max = 255, message = "Value too long")
    private String value; // "dark", "16", "off"
}

