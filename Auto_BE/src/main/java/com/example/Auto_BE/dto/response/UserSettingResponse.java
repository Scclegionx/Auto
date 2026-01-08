package com.example.Auto_BE.dto.response;

import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class UserSettingResponse {

    private String settingKey;
    private String value;
    private String defaultValue;
}

