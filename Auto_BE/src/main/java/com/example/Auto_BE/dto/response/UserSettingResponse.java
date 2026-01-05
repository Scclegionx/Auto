package com.example.Auto_BE.dto.response;

import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class UserSettingResponse {

    private String settingKey; // "theme", "font_size", "voice_support"
    private String value; // Giá trị hiện tại của user: "dark", "16", "off"
    private String defaultValue; // Giá trị mặc định từ Settings
}

