package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.DeviceTokenRequest;
import com.example.Auto_BE.dto.response.DeviceTokenResponse;
import com.example.Auto_BE.repository.DeviceTokenRepository;
import com.example.Auto_BE.service.DeviceTokenService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/device-token")
public class DeviceTokenController {
    private final DeviceTokenService deviceTokenService;

    public DeviceTokenController(DeviceTokenService deviceTokenService) {
        this.deviceTokenService = deviceTokenService;
    }

    @PostMapping("/register")
    public ResponseEntity<BaseResponse<DeviceTokenResponse>> registerDeviceToken(@RequestBody @Valid DeviceTokenRequest deviceTokenRequest,
                                                                                   Authentication authentication) {
        BaseResponse<DeviceTokenResponse> response = deviceTokenService.registerDeviceToken(deviceTokenRequest, authentication);
        return ResponseEntity.ok(response);
    }
}