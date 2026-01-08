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

    @PostMapping
    public ResponseEntity<BaseResponse<UserGuideResponse>> createUserGuide(
            @RequestPart("request") @Valid CreateUserGuideRequest request,
            @RequestPart("videoFile") MultipartFile videoFile) {
        BaseResponse<UserGuideResponse> response = userGuideService.createUserGuide(request, videoFile);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    @GetMapping("/{id}")
    public ResponseEntity<BaseResponse<UserGuideResponse>> getUserGuide(@PathVariable Long id) {
        BaseResponse<UserGuideResponse> response = userGuideService.getUserGuideById(id);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    @GetMapping("/elder")
    public ResponseEntity<BaseResponse<List<UserGuideResponse>>> getElderUserGuides() {
        BaseResponse<List<UserGuideResponse>> response = userGuideService.getElderUserGuides();
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    @GetMapping("/supervisor")
    public ResponseEntity<BaseResponse<List<UserGuideResponse>>> getSupervisorUserGuides() {
        BaseResponse<List<UserGuideResponse>> response = userGuideService.getSupervisorUserGuides();
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    @PutMapping("/{id}")
    public ResponseEntity<BaseResponse<UserGuideResponse>> updateUserGuide(
            @PathVariable Long id,
            @RequestPart("request") @Valid UpdateUserGuideRequest request,
            @RequestPart(value = "videoFile", required = false) MultipartFile videoFile) {
        BaseResponse<UserGuideResponse> response = userGuideService.updateUserGuide(id, request, videoFile);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<BaseResponse<String>> deleteUserGuide(@PathVariable Long id) {
        BaseResponse<String> response = userGuideService.deleteUserGuide(id);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }
}

