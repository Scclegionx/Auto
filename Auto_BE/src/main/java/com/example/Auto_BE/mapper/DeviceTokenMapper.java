package com.example.Auto_BE.mapper;

import com.example.Auto_BE.dto.request.DeviceTokenRequest;
import com.example.Auto_BE.dto.response.DeviceTokenResponse;
import com.example.Auto_BE.entity.DeviceToken;
import com.example.Auto_BE.entity.User;

public class DeviceTokenMapper {

    public static DeviceTokenResponse toResponse(DeviceToken deviceToken) {
        return DeviceTokenResponse.builder()
                .fcmToken(deviceToken.getFcmToken())
                .deviceId(deviceToken.getDeviceId())
                .deviceType(deviceToken.getDeviceType())
                .deviceName(deviceToken.getDeviceName())
                .isActive(deviceToken.getIsActive())
                .userId(deviceToken.getUser() != null ? deviceToken.getUser().getId() : null)
                .build();
    }
    public static DeviceToken toEntity(DeviceTokenRequest request, User user) {
        return new DeviceToken()
                .setFcmToken(request.getFcmToken())
                .setDeviceId(request.getDeviceId())
                .setDeviceType(request.getDeviceType())
                .setDeviceName(request.getDeviceName())
                .setIsActive(true)
                .setUser(user);
    }
}
