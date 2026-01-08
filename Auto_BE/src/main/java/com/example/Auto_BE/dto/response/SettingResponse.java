package com.example.Auto_BE.dto.response;

import com.example.Auto_BE.entity.enums.ESettingType;
import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SettingResponse {

    private Long id;
    private String settingKey;
    private String name;
    private String description;
    private String defaultValue;
    private String possibleValues;
    private Boolean isActive;
    private ESettingType settingType;
}

