package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.ChangePasswordRequest;
import com.example.Auto_BE.dto.request.UpdateProfileRequest;
import com.example.Auto_BE.service.UserService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/users")
public class UserController {
    private final UserService userService;

    public UserController(UserService userService) {
        this.userService = userService;
    }

    @GetMapping("/profile")
    public ResponseEntity<BaseResponse<?>> getUserProfile(Authentication authentication) {
        BaseResponse<?> response = userService.getUserProfile(authentication);
        return ResponseEntity.ok(response);
    }

    @PutMapping("/profile")
    public ResponseEntity<BaseResponse<?>> updateUserProfile(@RequestBody @Valid UpdateProfileRequest updateProfileRequest,
                                                             Authentication authentication) {
        BaseResponse<?> response = userService.updateUserProfile(updateProfileRequest, authentication);
        return ResponseEntity.ok(response);
    }

    @PutMapping("/change-password")
    public ResponseEntity<BaseResponse<String>> changePassword(@RequestBody @Valid ChangePasswordRequest changePasswordRequest,
                                                               Authentication authentication) {
        BaseResponse<String> response = userService.changePassword(changePasswordRequest, authentication);
        return ResponseEntity.ok(response);
    }
}