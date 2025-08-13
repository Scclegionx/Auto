package com.example.Auto_BE.dto.request;

import jakarta.validation.constraints.NotBlank;
import lombok.*;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DeviceTokenRequest {
    @NotBlank
    private String fcmToken;

    @NotBlank
    private String deviceId;

    @NotBlank
    private String deviceType;

    @NotBlank
    private String deviceName;

}