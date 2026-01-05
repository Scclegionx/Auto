package com.example.Auto_BE.dto.response;

import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SettingResponse {

    private Long id;
    private String settingKey; // "theme", "font_size", "voice_support"
    private String name; // "Nền", "Font Size", "Hỗ trợ nói"
    private String description;
    private String defaultValue; // Giá trị mặc định
    private String possibleValues; // Các giá trị có thể
    private Boolean isActive;
}

