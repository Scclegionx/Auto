package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.GetUserSettingsRequest;
import com.example.Auto_BE.dto.request.UpdateMultipleUserSettingsRequest;
import com.example.Auto_BE.dto.request.UpdateUserSettingRequest;
import com.example.Auto_BE.dto.response.AllUserSettingsResponse;
import com.example.Auto_BE.dto.response.SettingResponse;
import com.example.Auto_BE.service.SettingService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/settings")
public class SettingController {

    private final SettingService settingService;

    public SettingController(SettingService settingService) {
        this.settingService = settingService;
    }

    /**
     * Lấy tất cả các loại settings có sẵn (danh sách settings)
     * GET /api/settings
     */
    @GetMapping
    public ResponseEntity<BaseResponse<List<SettingResponse>>> getAllSettings() {
        
        System.out.println("API: Get all settings");
        
        BaseResponse<List<SettingResponse>> response = settingService.getAllSettings();
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Lấy tất cả giá trị settings của user
     * POST /api/settings/user
     * Body: { "userId": 1 } hoặc { "userId": null } cho GUEST
     */
    @PostMapping("/user")
    public ResponseEntity<BaseResponse<AllUserSettingsResponse>> getUserSettings(
            @RequestBody(required = false) GetUserSettingsRequest request) {
        
        System.out.println("API: Get user settings - userId: " + (request != null ? request.getUserId() : null));
        
        // Nếu không có request body, tạo request với userId = null (GUEST)
        if (request == null) {
            request = GetUserSettingsRequest.builder().userId(null).build();
        }
        
        BaseResponse<AllUserSettingsResponse> response = settingService.getUserSettings(request);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Cập nhật 1 setting của user
     * PUT /api/settings/user
     * Body: { "userId": 1, "settingKey": "theme", "value": "dark" }
     */
    @PutMapping("/user")
    public ResponseEntity<BaseResponse<com.example.Auto_BE.dto.response.UserSettingResponse>> updateUserSetting(
            @RequestBody @Valid UpdateUserSettingRequest request) {
        
        System.out.println("API: Update user setting - userId: " + request.getUserId() + ", key: " + request.getSettingKey());
        
        BaseResponse<com.example.Auto_BE.dto.response.UserSettingResponse> response = 
                settingService.updateUserSetting(request);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Cập nhật nhiều settings của user cùng lúc
     * PUT /api/settings/user/multiple
     * Body: { "userId": 1, "settings": [...] }
     */
    @PutMapping("/user/multiple")
    public ResponseEntity<BaseResponse<AllUserSettingsResponse>> updateMultipleUserSettings(
            @RequestBody @Valid UpdateMultipleUserSettingsRequest request) {
        
        System.out.println("API: Update multiple user settings - userId: " + request.getUserId() + ", count: " + request.getSettings().size());
        
        BaseResponse<AllUserSettingsResponse> response = 
                settingService.updateMultipleUserSettings(request);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Reset settings của user về giá trị mặc định
     * DELETE /api/settings/user/{userId}
     */
    @DeleteMapping("/user/{userId}")
    public ResponseEntity<BaseResponse<String>> resetUserSettingsToDefault(
            @PathVariable Long userId) {
        
        System.out.println("API: Reset user settings to default - userId: " + userId);
        
        BaseResponse<String> response = settingService.resetUserSettingsToDefault(userId);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }
}
