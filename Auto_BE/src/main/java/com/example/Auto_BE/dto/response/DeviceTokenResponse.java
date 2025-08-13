package com.example.Auto_BE.dto.response;

import lombok.*;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DeviceTokenResponse {
    private String fcmToken;
    private String deviceId;
    private String deviceType;
    private String deviceName;
    private Boolean isActive;
    private Long userId;
}