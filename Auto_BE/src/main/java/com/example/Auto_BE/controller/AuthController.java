package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.ForgotPasswordRequest;
import com.example.Auto_BE.dto.request.LoginRequest;
import com.example.Auto_BE.dto.request.RegisterRequest;
import com.example.Auto_BE.dto.request.ResendVerificationRequest;
import com.example.Auto_BE.dto.request.SendVerificationRequest;
import com.example.Auto_BE.dto.request.VerifyOtpRequest;
import com.example.Auto_BE.dto.response.LoginResponse;
import com.example.Auto_BE.service.AuthService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/auth")
public class AuthController {
    private final AuthService authService;

    public AuthController(AuthService authService) {
        this.authService = authService;
    }

    @PostMapping("/login")
    public ResponseEntity<BaseResponse<LoginResponse>> login(@RequestBody @Valid LoginRequest loginRequest) {
        BaseResponse<LoginResponse> response = authService.login(loginRequest);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/register")
    public ResponseEntity<BaseResponse<Void>> register(@RequestBody @Valid RegisterRequest registerRequest) {
        BaseResponse<Void> response = authService.register(registerRequest);
        return ResponseEntity.ok(response);
    }

    /**
     * API mới: Gửi mã OTP xác thực email (user chủ động gọi)
     */
    @PostMapping("/send-verification-otp")
    public ResponseEntity<BaseResponse<Void>> sendVerificationOtp(@RequestBody @Valid SendVerificationRequest request) {
        BaseResponse<Void> response = authService.sendVerificationOtp(request);
        return ResponseEntity.ok(response);
    }

    /**
     * API mới: Xác thực mã OTP
     */
    @PostMapping("/verify-otp")
    public ResponseEntity<BaseResponse<Void>> verifyOtp(@RequestBody @Valid VerifyOtpRequest request) {
        BaseResponse<Void> response = authService.verifyOtp(request);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/verify")
    public ResponseEntity<BaseResponse<Void>> verifyEmail(@RequestParam String token) {
        BaseResponse<Void> response = authService.verifyEmail(token);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/resend-verification")
    public ResponseEntity<BaseResponse<Void>> resendVerificationEmail(@RequestBody @Valid ResendVerificationRequest resendVerificationRequest) {
        BaseResponse<Void> response = authService.resendVerificationEmail(resendVerificationRequest);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/send-forgot-password-otp")
    public ResponseEntity<BaseResponse<Void>> sendForgotPasswordOtp(@RequestBody @Valid SendVerificationRequest request) {
        BaseResponse<Void> response = authService.sendForgotPasswordOtp(request);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/verify-forgot-password-otp")
    public ResponseEntity<BaseResponse<Void>> verifyForgotPasswordOtp(@RequestBody @Valid VerifyOtpRequest request) {
        BaseResponse<Void> response = authService.verifyForgotPasswordOtp(request);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/forgot-password")
    public ResponseEntity<BaseResponse<Void>> forgotPassword(@RequestBody @Valid ForgotPasswordRequest forgotPasswordRequest) {
        BaseResponse<Void> response = authService.forgotPassword(forgotPasswordRequest);
        return ResponseEntity.ok(response);
    }
}