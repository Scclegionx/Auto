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
    private String userType; // "ELDER", "SUPERVISOR", "GUEST"
    private List<UserSettingResponse> settings; // Danh sách tất cả settings của user
}

