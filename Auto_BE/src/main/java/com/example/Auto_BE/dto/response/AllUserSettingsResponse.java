package com.example.Auto_BE.dto.response;

import lombok.*;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class AllUserSettingsResponse {

    private Long userId;
    private String userType;
    private List<UserSettingResponse> settings;
}

