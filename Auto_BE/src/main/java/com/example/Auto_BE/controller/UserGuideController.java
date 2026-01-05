package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.CreateUserGuideRequest;
import com.example.Auto_BE.dto.request.UpdateUserGuideRequest;
import com.example.Auto_BE.dto.response.UserGuideResponse;
import com.example.Auto_BE.service.UserGuideService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@RestController
@RequestMapping("/api/user-guides")
public class UserGuideController {

    private final UserGuideService userGuideService;

    public UserGuideController(UserGuideService userGuideService) {
        this.userGuideService = userGuideService;
    }

    /**
     * Tạo user guide mới (có upload video)
     * POST /api/user-guides
     * Content-Type: multipart/form-data
     * Body: title, description, userType, thumbnailUrl (optional), displayOrder (optional), isActive (optional), videoFile
     */
    @PostMapping
    public ResponseEntity<BaseResponse<UserGuideResponse>> createUserGuide(
            @RequestPart("request") @Valid CreateUserGuideRequest request,
            @RequestPart("videoFile") MultipartFile videoFile) {
        
        System.out.println("API: Create user guide - " + request.getTitle());
        
        BaseResponse<UserGuideResponse> response = userGuideService.createUserGuide(request, videoFile);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Lấy user guide theo ID
     * GET /api/user-guides/{id}
     */
    @GetMapping("/{id}")
    public ResponseEntity<BaseResponse<UserGuideResponse>> getUserGuide(@PathVariable Long id) {
        
        System.out.println("API: Get user guide - " + id);
        
        BaseResponse<UserGuideResponse> response = userGuideService.getUserGuideById(id);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Lấy tất cả user guides của Elder (chỉ active)
     * GET /api/user-guides/elder
     */
    @GetMapping("/elder")
    public ResponseEntity<BaseResponse<List<UserGuideResponse>>> getElderUserGuides() {
        
        System.out.println("API: Get elder user guides");
        
        BaseResponse<List<UserGuideResponse>> response = userGuideService.getElderUserGuides();
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Lấy tất cả user guides của Supervisor (chỉ active)
     * GET /api/user-guides/supervisor
     */
    @GetMapping("/supervisor")
    public ResponseEntity<BaseResponse<List<UserGuideResponse>>> getSupervisorUserGuides() {
        
        System.out.println("API: Get supervisor user guides");
        
        BaseResponse<List<UserGuideResponse>> response = userGuideService.getSupervisorUserGuides();
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Update user guide
     * PUT /api/user-guides/{id}
     * Content-Type: multipart/form-data
     * Body: request (UpdateUserGuideRequest), videoFile (optional - chỉ gửi khi muốn thay đổi video)
     */
    @PutMapping("/{id}")
    public ResponseEntity<BaseResponse<UserGuideResponse>> updateUserGuide(
            @PathVariable Long id,
            @RequestPart("request") @Valid UpdateUserGuideRequest request,
            @RequestPart(value = "videoFile", required = false) MultipartFile videoFile) {
        
        System.out.println("API: Update user guide - " + id);
        
        BaseResponse<UserGuideResponse> response = userGuideService.updateUserGuide(id, request, videoFile);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Xóa user guide
     * DELETE /api/user-guides/{id}
     */
    @DeleteMapping("/{id}")
    public ResponseEntity<BaseResponse<String>> deleteUserGuide(@PathVariable Long id) {
        
        System.out.println("API: Delete user guide - " + id);
        
        BaseResponse<String> response = userGuideService.deleteUserGuide(id);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }
}

