package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.UpdateUserSettingRequest;
import com.example.Auto_BE.dto.response.AllUserSettingsResponse;
import com.example.Auto_BE.dto.response.SettingResponse;
import com.example.Auto_BE.dto.response.UserSettingResponse;
import com.example.Auto_BE.entity.enums.ESettingType;
import com.example.Auto_BE.service.SettingService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/settings")
public class SettingController {

    private final SettingService settingService;

    public SettingController(SettingService settingService) {
        this.settingService = settingService;
    }

    @GetMapping
    public ResponseEntity<BaseResponse<List<SettingResponse>>> getAllSettings() {
        BaseResponse<List<SettingResponse>> response = settingService.getAllSettings();
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    @GetMapping("/type/{settingType}")
    public ResponseEntity<BaseResponse<List<SettingResponse>>> getAllBySettingType(
            @PathVariable ESettingType settingType) {
        BaseResponse<List<SettingResponse>> response = settingService.getAllBySettingType(settingType);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    @GetMapping("/type")
    public ResponseEntity<BaseResponse<List<SettingResponse>>> getAllBySettingTypes(
            @RequestParam List<ESettingType> settingTypes) {
        BaseResponse<List<SettingResponse>> response = settingService.getAllBySettingTypes(settingTypes);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    @GetMapping("/user")
    public ResponseEntity<BaseResponse<AllUserSettingsResponse>> getUserSettings(
            Authentication authentication) {
        BaseResponse<AllUserSettingsResponse> response = settingService.getUserSettings(authentication);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    @PutMapping("/user")
    public ResponseEntity<BaseResponse<UserSettingResponse>> updateUserSetting(
            @RequestBody @Valid UpdateUserSettingRequest request,
            Authentication authentication) {
        BaseResponse<UserSettingResponse> response =
                settingService.updateUserSetting(request, authentication);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }
}
